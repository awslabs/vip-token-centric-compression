# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import datasets
import argparse
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import json
import pyarrow.dataset as ds
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from itertools import repeat
from collections import deque
import warnings
import pandas as pd
import os

num_docs_per_partition = 10000

# filter out dataset with token per byte > 0.3
filter_out = [
    "PubMed Central",
    "ArXiv",
    "Github",
    "StackExchange",
    "DM Mathematics",
    "Ubuntu IRC",
    "EuroParl",
    "YoutubeSubtitles",
    "Enron Emails",
]
filter_out = set(filter_out)
print(filter_out)

train_files = [os.path.join("../mnt/pile/train", file) for file in os.listdir("../mnt/pile/train") if file.endswith(".jsonl")]
val_file = "../mnt/pile/val.jsonl"
test_file = "../mnt/pile/test.jsonl"
print(train_files)
print(val_file)
print(test_file)

def filter_partition(files, output_folder, filter_out, num_docs_per_partition):
    os.makedirs(output_folder, exist_ok = True)
    partition = []
    partition_idx = 0
    for file in files:
        print(f"reading {file}")
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            doc = json.loads(line)
            if doc["meta"]["pile_set_name"] in filter_out:
                continue
            partition.append(doc)
            if len(partition) >= num_docs_per_partition:
                output_path = os.path.join(output_folder, f"{partition_idx:08d}.jsonl")
                print(f"outputing {output_path}")
                with open(output_path, "w") as out_f:
                    for doc in partition:
                        out_f.write(json.dumps(doc) + "\n")
                partition = []
                partition_idx += 1
        del lines
    if len(partition) != 0:
        output_path = os.path.join(output_folder, f"{partition_idx:08d}.jsonl")
        print(f"outputing {output_path}")
        with open(output_path, "w") as out_f:
            for doc in partition:
                out_f.write(json.dumps(doc) + "\n")
        partition = []
        partition_idx += 1

filter_partition([val_file], "../mnt/filtered_pile/val", filter_out, num_docs_per_partition)
filter_partition([test_file], "../mnt/filtered_pile/test", filter_out, num_docs_per_partition)
filter_partition(train_files, "../mnt/filtered_pile/train", filter_out, num_docs_per_partition)
