# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from src.args import Args, Options, import_config
import os

__common = import_config(os.path.join(os.path.dirname(os.path.realpath(__file__)), "common.py"))
seed = __common.seed
output_dir = __common.output_dir
save_top_k = __common.save_top_k
project = __common.project
trainer = __common.trainer
optimizer = __common.optimizer
data = __common.data
model = Args()
model.pl_module = "src.t5.tasks.downstream.DownstreamModelModule"
model.max_decoder_length = data.collator_args.max_decoder_length

trainer.precision = 16
data.tokenizer = "allenai/led-base-16384"

model.model_type = "transformers.LEDForConditionalGeneration"
model.pretrained_config = "allenai/led-base-16384"
model.load_pretrain = {
    "model_type":"transformers.LEDForConditionalGeneration",
    "checkpoint":"allenai/led-base-16384"
}
optimizer.base_learning_rate = 3e-5
