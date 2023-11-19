# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

dataset = load_dataset('scientific_papers', 'arxiv')
print(dataset)
dataset["train"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/arxiv/train")
dataset["validation"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/arxiv/validation")

dataset = load_dataset('scientific_papers', 'pubmed')
print(dataset)
dataset["train"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/pubmed/train")
dataset["validation"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/pubmed/validation")
