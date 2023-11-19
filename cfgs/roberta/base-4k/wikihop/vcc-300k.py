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
data.collator_args.encode_type = "gsop"

model = Args()
model.pl_module = __common.model.pl_module
model.encoder_type = "src.roberta.models.vcc.VccRobertaModel"
# model.load_pretrain = {
#     "model_type":"src.roberta.tasks.mlm.RobertaForMLM",
#     "checkpoint":"hf_ckpts/roberta/base-4k/mra-2n-cpmlm-300k.py/epoch=00000-step=00300000"
# }
model.initializer_range = 0.02
model.vocab_size = 50265
model.type_vocab_size = 128
model.pad_token_id = 1
model.num_hidden_layers = 12
model.num_attention_heads = 12
model.layer_norm_eps = 1e-05
model.intermediate_size = 3072
model.hidden_size = 768
model.hidden_dropout_prob = 0.1
model.attention_probs_dropout_prob = 0.1
model.max_position_embeddings = 4096

model.include_positions_in_mixed_states = False
model.short_segment_size = 512
model.encoder_layer_types = ["src.roberta.models.prenorm.EncoderLayer"] * 4 + ["src.roberta.models.vcc.VccEncoderLayer"] * 8

model.coarser_type = "mean"
model.selector_type = "attention_based_selector"
model.explore_prob = 0.1
model.scale_correction = True

model.block_size = 16
model.num_important_tokens = 16 * model.block_size
model.num_fine_blocks = 80
