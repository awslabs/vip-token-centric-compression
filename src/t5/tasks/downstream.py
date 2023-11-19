# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pytorch_lightning as pl
import torch
import os
import json
import time
import datasets
import evaluate
from .pretrain import PretrainModelModule
from collections import defaultdict
from rouge_score import rouge_scorer
from src.utils import filter_inputs

class DownstreamModelModule(PretrainModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)
        self.squad_metric = evaluate.load('squad')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer = True, split_summaries = True)
        self.best_metrics = {}

    def generate(self, batch):
        generated_output = self.model.generate(
            batch["input_ids"],
            attention_mask = batch["attention_mask"],
            max_length = self.config.model.max_decoder_length
        )
        return generated_output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        filtered = filter_inputs(self.model.forward, batch)
        output = self.model(**filtered)
        output = {"loss":output.loss}
        generated_output = self.generate(batch)

        labels = batch["non_truncated_labels"]

        labels[labels == -100] = self.tokenizer.pad_token_id
        squad_scores = {"generation_em":[], "generation_f1":[]}
        rouge_scores = {"rouge1":[], "rouge2":[], "rougeL":[], "rougeLsum":[]}
        for b in range(labels.shape[0]):
            ground_truth = self.tokenizer.decode(labels[b].tolist(), skip_special_tokens = True)
            generation = self.tokenizer.decode(generated_output[b].tolist(), skip_special_tokens = True)

            metrics = self.squad_metric.compute(
                references = [{'answers': {'answer_start': [0], 'text': [ground_truth]}, 'id': str(b)}],
                predictions = [{'prediction_text': generation, 'id': str(b)}]
            )
            squad_scores["generation_em"].append(metrics["exact_match"])
            squad_scores["generation_f1"].append(metrics["f1"])

            scores = self.rouge_scorer.score(ground_truth, generation)
            rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
            rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
            rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
            rouge_scores["rougeLsum"].append(scores["rougeLsum"].fmeasure)

        for key, val in squad_scores.items():
            output[key] = torch.tensor(val, device = generated_output.device).float().mean()
        for key, val in rouge_scores.items():
            output[key] = torch.tensor(val, device = generated_output.device).float().mean()

        for key, val in self.sync_dict(output).items():
            self.log(f"val.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def validation_epoch_end(self, outputs):
        summary = defaultdict(list)
        for output in outputs:
            for key, val in self.sync_dict(output).items():
                summary[key].append(val.item())
        summary = {key:sum(val)/float(len(val)) for key, val in summary.items()}
        for key in summary:
            if f"{key}-low" not in self.best_metrics:
                self.best_metrics[f"{key}-low"] = summary[key]
                self.best_metrics[f"{key}-high"] = summary[key]
            else:
                self.best_metrics[f"{key}-low"] = min(self.best_metrics[f"{key}-low"], summary[key])
                self.best_metrics[f"{key}-high"] = max(self.best_metrics[f"{key}-high"], summary[key])

        for key, val in self.best_metrics.items():
            self.log(f"val.best.{key}", val, prog_bar = True, logger = True)
