# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FilteringPress — online per-token keep/skip decisions during decoding.
"""

import pytest
import torch
from transformers import DynamicCache, pipeline

from kvpress import FilteringPress, KeyDiffPress, KnormPress, PrefillDecodingPress, StreamingLLMPress, TOVAPress


@pytest.fixture(scope="module")
def pipe():
    return pipeline("kv-press-text-generation", model="MaxJeblick/llama2-0b-unit-test", device_map="auto")


CONTEXT = "The quick brown fox jumps over the lazy dog. " * 10
QUESTION = "What animal jumps over the dog?"


def test_filtering_press_reduces_cache(pipe):
    """FilteringPress should produce a smaller cache than no compression."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_baseline = DynamicCache()
    with torch.no_grad():
        model.generate(input_ids, past_key_values=cache_baseline, max_new_tokens=20, do_sample=False)
    baseline_len = cache_baseline.get_seq_length()

    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.9)
    cache_filtered = DynamicCache()
    with torch.no_grad(), press(model):
        model.generate(input_ids, past_key_values=cache_filtered, max_new_tokens=20, do_sample=False)
    filtered_len = cache_filtered.get_seq_length()

    assert (
        filtered_len < baseline_len
    ), f"filtered cache ({filtered_len}) should be smaller than baseline ({baseline_len})"


def test_filtering_press_no_op_at_zero_ratio(pipe):
    """target_compression_ratio=0 should not filter any tokens."""
    cache_baseline = DynamicCache()
    pipe(CONTEXT, question=QUESTION, cache=cache_baseline, max_new_tokens=20)

    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.0)
    cache_filtered = DynamicCache()
    pipe(CONTEXT, question=QUESTION, press=press, cache=cache_filtered, max_new_tokens=20)

    for layer_idx in range(len(cache_baseline.layers)):
        assert cache_baseline.layers[layer_idx].keys.shape[2] == cache_filtered.layers[layer_idx].keys.shape[2]


def test_filtering_press_with_prefill_decoding(pipe):
    """FilteringPress should work as decoding_press inside PrefillDecodingPress."""
    combined_press = PrefillDecodingPress(
        prefilling_press=KeyDiffPress(compression_ratio=0.5),
        decoding_press=FilteringPress(base_press=KeyDiffPress(), target_compression_ratio=0.5),
    )

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=combined_press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, "No answer generated"


@pytest.mark.parametrize("scorer_cls", [KnormPress, KeyDiffPress, TOVAPress, StreamingLLMPress])
def test_filtering_press_with_different_scorers(pipe, scorer_cls):
    """FilteringPress should work with any ScorerPress."""
    press = FilteringPress(base_press=scorer_cls(), target_compression_ratio=0.5)

    cache = DynamicCache()
    result = pipe(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=15)

    assert len(result["answer"]) > 0, f"No answer generated with {scorer_cls.__name__}"


def test_filtering_press_higher_ratio_filters_more(pipe):
    """Higher compression ratio should produce a smaller cache."""
    model = pipe.model
    tokenizer = pipe.tokenizer
    device = model.device

    input_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)

    cache_low = DynamicCache()
    press_low = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.3)
    with torch.no_grad(), press_low(model):
        model.generate(input_ids, past_key_values=cache_low, max_new_tokens=20, do_sample=False)

    cache_high = DynamicCache()
    press_high = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.7)
    with torch.no_grad(), press_high(model):
        model.generate(input_ids, past_key_values=cache_high, max_new_tokens=20, do_sample=False)

    low_len = cache_low.get_seq_length()
    high_len = cache_high.get_seq_length()
    assert high_len <= low_len, f"higher ratio cache ({high_len}) should be <= lower ratio cache ({low_len})"


def test_filtering_press_reuse_across_sequences(pipe):
    """Reusing a FilteringPress across sequences should not crash."""
    press = FilteringPress(base_press=KnormPress(), target_compression_ratio=0.5)

    model = pipe.model
    device = model.device
    long_ids = torch.arange(1, 81, dtype=torch.long, device=device).unsqueeze(0)
    short_ids = torch.arange(1, 9, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad(), press(model):
        model.generate(long_ids, max_new_tokens=6, do_sample=False)
        model.generate(short_ids, max_new_tokens=6, do_sample=False)
