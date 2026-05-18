# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from json import load
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.block_press import BlockPress
from kvpress.presses.compression_ratio_decoding_press import CompressionRatioDecodingPress
from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress

KEYDIFF_BLOCK_SIZE = 128


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def load_config() -> dict:
    config_path = Path(__file__).with_name("my_script_config.json")
    with config_path.open() as file:
        return load(file)

def make_keydiff_press(
    compression_ratio: float,
    apply_keydiff_during_prefill: bool,
    apply_keydiff_during_generation: bool,
):
    if compression_ratio == 0.0 or (
        not apply_keydiff_during_prefill and not apply_keydiff_during_generation
    ):
        return None

    prefill_press = (
        BlockPress(
            press=KeyDiffPress(compression_ratio=compression_ratio),
            block_size=KEYDIFF_BLOCK_SIZE,
        )
        if apply_keydiff_during_prefill
        else None
    )
    if not apply_keydiff_during_generation:
        return prefill_press

    decoding_press = CompressionRatioDecodingPress(
        base_press=KeyDiffPress(),
        compression_interval=8,
        target_compression_ratio=compression_ratio,
    )

    if not apply_keydiff_during_prefill:
        return decoding_press

    return PrefillDecodingPress(
        prefilling_press=prefill_press,
        decoding_press=decoding_press,
    )


def main() -> None:
    config = load_config()
    device = pick_device()
    dtype = pick_dtype(device)
    hf_token = get_hf_token()
    model_names = config["models"]

    print(f"Using device: {device}")

    context = config["context"]
    questions = config["questions"]
    compression_ratios = config["compression_ratios"]
    max_new_tokens = config["max_new_tokens"]
    apply_keydiff_during_prefill = config["apply_keydiff_during_prefill"]
    apply_keydiff_during_generation = config["apply_keydiff_during_generation"]

    print("\nQuestions:\n")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question}")

    for model_name in model_names:
        print(f"\nLoading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, token=hf_token).to(device).eval()
        pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)

        for compression_ratio in compression_ratios:
            for question in questions:
                press = make_keydiff_press(
                    compression_ratio,
                    apply_keydiff_during_prefill,
                    apply_keydiff_during_generation,
                )
                result = pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)

                print(f"\nModel: {model_name}")
                print(f"Compression ratio: {compression_ratio}")
                print(f"Question: {question}")
                print(result["answer"])


if __name__ == "__main__":
    main()
