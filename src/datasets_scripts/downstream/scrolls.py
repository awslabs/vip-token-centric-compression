# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

for name in ["gov_report", "summ_screen_fd", "qmsum", "narrative_qa", "qasper", "quality", "contract_nli"]:
    dataset = load_dataset("tau/scrolls", name)
    print(dataset)
    dataset["train"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/scrolls-{name}/train")
    dataset["validation"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/scrolls-{name}/validation")
    dataset["test"].save_to_disk(f"src/datasets_scripts/downstream/arrow_output/scrolls-{name}/test")
