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
from .mlm import MLMCollator

class PermutedMLMCollator(MLMCollator):

    def __init__(
            self,
            tokenizer,
            num_masked_tokens,
            max_sequence_length,
            num_segments_range,
            max_num_segments,
        ):
        super().__init__(tokenizer, num_masked_tokens, max_sequence_length)

        assert len(num_segments_range) == 2 and num_segments_range[0] <= num_segments_range[1]
        self.num_segments_range = num_segments_range
        self.max_num_segments = max_num_segments

    def random_truncate(self, sentence_lengths, max_sequence_length):
        # randomly select truncation points
        truncate_start = 0
        truncate_candidates = []
        accumulated_lengths = 0
        for sent_idx in reversed(range(len(sentence_lengths))):
            accumulated_lengths += sentence_lengths[sent_idx]
            if accumulated_lengths >= max_sequence_length:
                truncate_candidates.append(sent_idx)
        if len(truncate_candidates) > 0:
            truncate_start = random.choice(truncate_candidates)

        truncate_end = len(sentence_lengths)
        accumulated_lengths = 0
        for sent_idx in range(truncate_start, len(sentence_lengths)):
            accumulated_lengths += sentence_lengths[sent_idx]
            if accumulated_lengths >= max_sequence_length:
                truncate_end = sent_idx + 1
                break

        return truncate_start, truncate_end

    def get_segments(self, sequence):
        assert len(sequence) > 1
        tokenizer = self.tokenizer

        # sample the target number of segments for the sequence
        target_num_segments = random.randint(self.num_segments_range[0], self.num_segments_range[1])
        adjusted_max_sequence_length = self.max_sequence_length - 2 * target_num_segments

        doc_separator = "".join([tokenizer.eos_token] * 2)
        sent_separator = tokenizer.eos_token

        documents = []
        for doc in sequence.split(doc_separator):
            doc = doc.strip()
            if len(doc) != 0:
                documents.append(doc)

        # segment document into sentences
        sentences = []
        rough_sentence_lengths = []
        document_ids = []
        for doc_idx, doc in enumerate(documents):
            for sent in doc.split(sent_separator):
                if len(sent.strip()) == 0:
                    continue
                sentences.append(sent)
                rough_sentence_lengths.append(len(sent.split(" ")))
                document_ids.append(doc_idx)

        # do a rough truncation to save computation on tokenization
        truncate_start, truncate_end = self.random_truncate(rough_sentence_lengths, adjusted_max_sequence_length)
        sentences = sentences[truncate_start:truncate_end]
        document_ids = document_ids[truncate_start:truncate_end]

        tokenized_sentences = []
        exact_sentence_lengths = []
        for sent in sentences:
            sent = tokenizer.encode(sent, add_special_tokens = False)
            tokenized_sentences.append(sent)
            exact_sentence_lengths.append(len(sent))

        truncate_start, truncate_end = self.random_truncate(exact_sentence_lengths, adjusted_max_sequence_length)
        tokenized_sentences = tokenized_sentences[truncate_start:truncate_end]
        document_ids = document_ids[truncate_start:truncate_end]

        assert len(tokenized_sentences) == len(document_ids)

        # add segment breakpoints on document boundaries
        breakpoints = []
        last_doc_id = document_ids[0]
        accumulated_lengths = 0
        for sent_idx in range(len(tokenized_sentences)):
            if document_ids[sent_idx] != last_doc_id:
                breakpoints.append(accumulated_lengths)
                last_doc_id = document_ids[sent_idx]
            accumulated_lengths += len(tokenized_sentences[sent_idx])
        breakpoints.append(accumulated_lengths)

        # add rough segmentation points if requires more segments
        # used for sentence order predictions
        # TODO: find a better way to select breakpoints
        if target_num_segments > len(breakpoints):
            extra_num_bps = target_num_segments - len(breakpoints)
            breakpoints.extend((np.random.rand(extra_num_bps) * accumulated_lengths).astype(int).tolist())
            breakpoints = sorted(breakpoints)
            assert target_num_segments == len(breakpoints)

        # iterate through documents and sentences
        # if reach breakpoints, do segmentation
        segments = []
        curr_segment = []
        doc_ids = []
        accumulated_lengths = 0
        curr_breakpoint_idx = 0
        check_doc_id = document_ids[0]
        for sent_idx in range(len(tokenized_sentences)):
            sent = tokenized_sentences[sent_idx]
            doc_id = document_ids[sent_idx]
            curr_segment.extend(sent)
            accumulated_lengths += len(sent)
            assert check_doc_id == doc_id

            # if current breakpoint is reached, move to next breakpoint and start a new segment
            if accumulated_lengths >= breakpoints[curr_breakpoint_idx]:
                curr_breakpoint_idx += 1
                segments.append(curr_segment)
                doc_ids.append(doc_id)
                curr_segment = []
                if sent_idx + 1 < len(document_ids):
                    check_doc_id = document_ids[sent_idx + 1]
                else:
                    check_doc_id = None
                if len(doc_ids) >= self.max_num_segments:
                    print("Encounter len(doc_ids) >= self.max_num_segments")
                    break

        assert len(curr_segment) == 0

        return segments, doc_ids

    def process_one_sequence(self, sequence):

        tokenizer = self.tokenizer
        segments, doc_ids = self.get_segments(sequence)

        # randomly permute the segments
        permutation = np.argsort(np.random.rand(len(segments))).tolist()

        # construct inputs
        sequence = []
        segment_ids = []
        sequence_mask = []
        segment_lengths = []

        for idx, perm_index in enumerate(permutation):

            segment = [tokenizer.convert_tokens_to_ids(tokenizer.cls_token)]
            segment.extend(segments[perm_index])
            segment.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))

            sequence.extend(segment)
            segment_ids.extend([idx] * len(segment))
            sequence_mask.extend([1] * len(segment))
            segment_lengths.append(len(segment))

            if len(sequence) >= self.max_sequence_length:
                break

        pos_ids = list(range(len(sequence)))

        # truncate or pad sequence to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
            pos_ids = pos_ids[:self.max_sequence_length]
            segment_ids = segment_ids[:self.max_sequence_length]
            sequence_mask = sequence_mask[:self.max_sequence_length]

        while len(sequence) < self.max_sequence_length:
            sequence.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            pos_ids.append(0)
            segment_ids.append(0)
            sequence_mask.append(0)

        special_tokens_mask = tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens = True)

        sequence = torch.tensor(sequence, dtype = torch.long)
        pos_ids = torch.tensor(pos_ids, dtype = torch.long)
        segment_ids = torch.tensor(segment_ids, dtype = torch.long)
        sequence_mask = torch.tensor(sequence_mask, dtype = torch.long)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype = torch.long)

        instance = {
            "sequence":sequence,
            "special_tokens_mask":special_tokens_mask,
            "position_ids":pos_ids,
            "token_type_ids":segment_ids,
            "attention_mask":sequence_mask,
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

        mlm_sequence, mlm_labels, masked_token_indices = self.mlm_masking(batch["sequence"], batch["special_tokens_mask"])
        del batch["sequence"]
        del batch["special_tokens_mask"]
        batch["importance_mask"] = (mlm_labels != -100).long()
        batch["masked_token_indices"] = masked_token_indices
        batch["input_ids"] = mlm_sequence
        batch["labels"] = mlm_labels

        return batch
