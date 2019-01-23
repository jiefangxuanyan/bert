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
                          input_fn_builder)

app = Flask(__name__)
if not app.config.from_envvar("BERT_FLASK_SETTINGS"):
    raise RuntimeError


def init_wsgi():
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(app.config["do_lower_case"],
                                                  app.config["init_checkpoint"])

    bert_config = modeling.BertConfig.from_json_file(app.config["bert_config_file"])

    if app.config["max_seq_length"] > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (app.config["max_seq_length"], bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(app.config["output_dir"])

    processor = WikiJoinProcessor()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=app.config["vocab_file"], do_lower_case=app.config["do_lower_case"])

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=app.config["output_dir"],
        save_checkpoints_steps=app.config["save_checkpoints_steps"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=app.config["iterations_per_loop"],
            num_shards=app.config["num_tpu_cores"],
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=app.config["init_checkpoint"],
        learning_rate=app.config["learning_rate"],
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

    @app.route("/", methods=["post"])
    def handler():
        text = request.get_data(cache=False, as_text=True)
        example = InputExample(guid="%s-%s" % ("test", example_id[0]), text_a=text, label="full")
        features = convert_examples_to_features([example], label_list, app.config["max_seq_length"],
                                                tokenizer)
        input_fn = input_fn_builder(features, app.config["max_seq_length"], False, False)
        result = estimator.predict(input_fn=input_fn)
        prediction, = result
        difficulty = prediction["difficulty"]
        example_id[0] += 1
        return Response(str(difficulty), mimetype="text/plain")


init_wsgi()
