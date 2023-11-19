# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

dataset = load_dataset('glue', 'mnli')
print(dataset)

dataset["train"].save_to_disk(f"/nobackup/zhanpeng/glue-mnli/train")
dataset["validation_matched"].save_to_disk(f"/nobackup/zhanpeng/glue-mnli/validation_matched")
dataset["validation_mismatched"].save_to_disk(f"/nobackup/zhanpeng/glue-mnli/validation_mismatched")
