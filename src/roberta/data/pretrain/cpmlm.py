# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoTokenizer
import random
import numpy as np
import torch
from .pmlm import PermutedMLMCollator

def sample_gumbel(vals_to_add, eps = 1e-10):
    u = torch.rand_like(vals_to_add)
    gs = -torch.log(-torch.log(u + eps) + eps)
    return gs

class ClusterPermutedMLMCollator(PermutedMLMCollator):

    def __init__(
            self,
            tokenizer,
            num_masked_tokens,
            max_sequence_length,
            num_segments_range,
            max_num_segments,
            num_masked_clusters_range,
            masked_cluster_sigma_range,
        ):
        super().__init__(tokenizer, num_masked_tokens, max_sequence_length, num_segments_range, max_num_segments)
        self.num_masked_clusters_range = num_masked_clusters_range
        self.masked_cluster_sigma_range = masked_cluster_sigma_range

    def mlm_masking(self, sequence, special_tokens_mask, masking_probs):
        tokenizer = self.tokenizer
        assert len(sequence.shape) == 2
        batch_size, sequence_length = sequence.shape

        inputs = sequence.clone()
        labels = sequence.clone()
        batch_indices = torch.arange(batch_size)[:, None]

        masked_token_noise = (masking_probs + 1e-6).log() + sample_gumbel(masking_probs) - 1000 * special_tokens_mask.float()
        masked_token_ranking = torch.argsort(masked_token_noise, descending = True, dim = -1)
        masked_token_mask = torch.zeros_like(masked_token_ranking)
        masked_token_mask[batch_indices, masked_token_ranking[:, :self.num_masked_tokens]] = 1

        masked_token_indices = torch.sort(masked_token_mask, descending = True, stable = True, dim = -1).indices
        masked_token_indices = masked_token_indices[:, :self.num_masked_tokens]

        masked_token_mask = masked_token_mask.bool()

        labels[~masked_token_mask] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_token_mask
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_token_mask & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype = torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels, masked_token_indices

    def sample_masking_probs(self, segments):
        num_masked_clusters = random.randint(self.num_masked_clusters_range[0], self.num_masked_clusters_range[1])
        masked_cluster_sigma = random.uniform(self.masked_cluster_sigma_range[0], self.masked_cluster_sigma_range[1])
        sequence_length = sum([len(seg) for seg in segments])
        clusters = torch.randint(sequence_length, size = (num_masked_clusters, ), dtype = torch.float)
        positions = torch.arange(sequence_length, dtype = torch.float)
        logits = - (positions[:, None] - clusters[None, :]) ** 2 / (2 * masked_cluster_sigma ** 2)
        probs = torch.exp(logits).mean(dim = -1).tolist()
        masking_probs = []
        start_idx = 0
        for seg in segments:
            masking_probs.append(probs[start_idx:(start_idx + len(seg))])
            start_idx = start_idx + len(seg)
        return masking_probs

    def process_one_sequence(self, sequence):

        tokenizer = self.tokenizer
        segments, doc_ids = self.get_segments(sequence)

        segmented_masking_probs = self.sample_masking_probs(segments)

        # randomly permute the segments
        permutation = np.argsort(np.random.rand(len(segments))).tolist()

        # construct inputs
        sequence = []
        masking_probs = []
        segment_ids = []
        sequence_mask = []
        segment_lengths = []

        for idx, perm_index in enumerate(permutation):

            segment = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)]
            segment.extend(segments[perm_index])
            segment.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))

            segment_masking_probs = [0] + segmented_masking_probs[perm_index] + [0]

            sequence.extend(segment)
            masking_probs.extend(segment_masking_probs)

            segment_ids.extend([idx] * len(segment))
            sequence_mask.extend([1] * len(segment))
            segment_lengths.append(len(segment))

            if len(sequence) >= self.max_sequence_length:
                break

        pos_ids = list(range(len(sequence)))

        # truncate or pad sequence to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
            masking_probs = masking_probs[:self.max_sequence_length]
            pos_ids = pos_ids[:self.max_sequence_length]
            segment_ids = segment_ids[:self.max_sequence_length]
            sequence_mask = sequence_mask[:self.max_sequence_length]

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            masking_probs.append(0)
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)

        special_tokens_mask = tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens = True)

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype = torch.long)
        masking_probs = torch.tensor(masking_probs, dtype = torch.float)

        instance = {
            "sequence":sequence,
            "special_tokens_mask":special_tokens_mask,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
            "masking_probs":masking_probs,
        }

        return instance

    def __call__(self, list_strs):
        # list_strs is a list of batch_size strings

        batch = {}
        for sequence in list_strs:
            instance = self.process_one_sequence(sequence)
            for key in instance:
                if key not in batch:
                    batch[key] = []
                batch[key].append(instance[key])

        batch = {key:torch.stack(batch[key], dim = 0) for key in batch}

        mlm_sequence, mlm_labels, masked_token_indices = self.mlm_masking(batch["sequence"], batch["special_tokens_mask"], batch["masking_probs"])
        del batch["sequence"]
        del batch["special_tokens_mask"]
        del batch["masking_probs"]
        batch["importance_mask"] = (mlm_labels != -100).long()
        batch["masked_token_indices"] = masked_token_indices
        batch["input_ids"] = mlm_sequence
        batch["labels"] = mlm_labels

        return batch
