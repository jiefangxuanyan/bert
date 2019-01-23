# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from flask import Flask, request, Response

import modeling
import tokenization
from run_wikijoin import (WikiJoinProcessor, model_fn_builder, InputExample, convert_examples_to_features,
                          input_fn_builder, convert_single_example)

app = Flask(__name__)
if not app.config.from_envvar("BERT_FLASK_SETTINGS"):
    raise RuntimeError


class StoredIterator:
    value = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.value


def generate_from_iterator(iterator):
    for feature in iterator:
        yield {
            "input_ids": feature.input_ids,
            "input_mask": feature.input_mask,
            "segment_ids": feature.segment_ids,
            "label_id": feature.label_id
        }


def online_input_fn_builder(iterator, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.Dataset.from_generator(generate_from_iterator, output_types={
            "input_ids": tf.int32,
            "input_mask": tf.int32,
            "segment_ids": tf.int32,
            "label_id": tf.int32
        }, output_shapes={
            "input_ids": [seq_length],
            "input_mask": [seq_length],
            "segment_ids": [seq_length],
            "label_id": [seq_length]
        }, args=(iterator,))

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def init_wsgi():
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(app.config["DO_LOWER_CASE"],
                                                  app.config["INIT_CHECKPOINT"])

    bert_config = modeling.BertConfig.from_json_file(app.config["BERT_CONFIG_FILE"])

    if app.config["MAX_SEQ_LENGTH"] > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (app.config["MAX_SEQ_LENGTH"], bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(app.config["OUTPUT_DIR"])

    processor = WikiJoinProcessor()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=app.config["VOCAB_FILE"], do_lower_case=app.config["DO_LOWER_CASE"])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=app.config["OUTPUT_DIR"],
        save_checkpoints_steps=app.config["SAVE_CHECKPOINTS_STEPS"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=app.config["ITERATIONS_PER_LOOP"],
            num_shards=app.config["NUM_TPU_CORES"],
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=app.config["INIT_CHECKPOINT"],
        learning_rate=app.config["LEARNING_RATE"],
        num_train_steps=None,
        num_warmup_steps=None,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=1)

    example_id = [5]

    stored_iterator = StoredIterator()

    input_fn = online_input_fn_builder(stored_iterator, app.config["MAX_SEQ_LENGTH"], False, False)
    result = estimator.predict(input_fn=input_fn)
    result_iterator = iter(result)

    @app.route("/eval", methods=["post"])
    def handler():
        text = request.get_data(cache=False, as_text=True)
        example = InputExample(guid="%s-%s" % ("test", example_id[0]), text_a=text, label="full")
        feature = convert_single_example(example_id[0], example, label_list,
                                         app.config["MAX_SEQ_LENGTH"], tokenizer)
        stored_iterator.value = feature
        prediction = next(result_iterator)
        difficulty = prediction["difficulty"]
        example_id[0] += 1
        return Response(str(difficulty), mimetype="text/plain")


init_wsgi()
