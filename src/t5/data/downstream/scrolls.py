# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from .base import BaseCollator

class GovReportCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["input"]
        output = instance["output"]
        return query, context, output

class SummScreenFDCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["input"]
        output = instance["output"]
        return query, context, output

class QmsumCollator(BaseCollator):
    def get_query_context_output(self, instance):
        inp = instance["input"]
        sep = "\n\n"
        segments = inp.split(sep)
        assert len(segments) == 2
        query = segments[0]
        context = segments[1]
        output = instance["output"]
        return query, context, output

class NarrativeQACollator(BaseCollator):
    def get_query_context_output(self, instance):
        inp = instance["input"]
        sep = "\n\n"
        segments = inp.split(sep)
        assert len(segments) >= 2
        query = segments[0]
        context = sep.join(segments[1:])
        output = instance["output"]
        return query, context, output

class QasperCollator(BaseCollator):
    def get_query_context_output(self, instance):
        inp = instance["input"]
        sep = "\n\n"
        segments = inp.split(sep)
        assert len(segments) >= 2
        query = segments[0]
        context = sep.join(segments[1:])
        output = instance["output"]
        return query, context, output

class ContractNLICollator(BaseCollator):
    def get_query_context_output(self, instance):
        inp = instance["input"]
        sep = "\n\n"
        segments = inp.split(sep)
        assert len(segments) >= 2
        query = segments[0]
        context = sep.join(segments[1:])
        output = instance["output"]
        return query, context, output

class QualityCollator(BaseCollator):
    def get_query_context_output(self, instance):
        inp = instance["input"]
        sep = "\n"
        segments = inp.split(sep)
        for i in range(len(segments)):
            if " (D) " in segments[i]:
                break
        assert i < len(segments)
        query = sep.join(segments[:(i+1)])
        context = sep.join(segments[(i+1):])
        assert " (A) " in query, query
        assert " (B) " in query, query
        assert " (C) " in query, query
        assert " (D) " in query, query

        output = instance["output"]
        return query, context, output
