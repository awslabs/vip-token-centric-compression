# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from .base import BaseCollator

class ArxivCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["article"]
        output = instance["abstract"]
        return query, context, output

class PubmedCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["article"]
        output = instance["abstract"]
        return query, context, output

class BigPatentCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["description"]
        output = instance["abstract"]
        return query, context, output

class CNNDailyMailCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["article"]
        output = instance["highlights"]
        return query, context, output

class MediaSumCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = "\n\n".join(instance["document"])
        output = instance["summary"]
        return query, context, output

class MultiNewsCollator(BaseCollator):
    def get_query_context_output(self, instance):
        query = "summarize"
        context = instance["document"]
        output = instance["summary"]
        return query, context, output
