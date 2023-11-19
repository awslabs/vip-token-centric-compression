# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

dataset = load_dataset('ccdv/mediasum', 'list')
print(dataset)
dataset["train"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/mediasum/train")
dataset["validation"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/mediasum/validation")
