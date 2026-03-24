# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import sys
import time
from typing import Literal, Optional

import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_request import InferenceRequest
from nemo.collections.llm import inference
from nemo.utils import logging


CheckpointFormats = Literal["torch_dist", "zarr"]


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()

    # generation args:
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt to generate text from Evo2. Defaults to a phylogenetic lineage tag for E coli. "
        "Ignored when --prompts-file is set.",
    )
    ap.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a file with one prompt per line. Loads the model once and runs each prompt in order. "
        "Requires --output-files-file; optional --seed-offsets (one integer per prompt, comma-separated).",
    )
    ap.add_argument(
        "--output-files-file",
        type=str,
        default=None,
        help="Path to a file with one output filename per line, aligned with --prompts-file.",
    )
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, help="Path to checkpoint directory containing pre-trained Evo2 model."
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature during sampling for generation.")
    ap.add_argument("--top-k", type=int, default=0, help="Top K during sampling for generation.")
    ap.add_argument("--top-p", type=float, default=0.0, help="Top P during sampling for generation.")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    ap.add_argument(
        "--seed-offsets",
        type=str,
        default=None,
        help="Comma-separated integer offsets, one per line in --prompts-file. "
        "Run j (1..num-runs) for prompt i uses seed = --seed + offset_i + j. Ignored without --prompts-file.",
    )
    ap.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of inference runs with the same prompt. Model is loaded once. Defaults to 1.",
    )
    # compute args:
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    # output args:
    ap.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file containing the generated text produced by the Evo2 model. If not provided, the output will be logged.",
    )
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )
    ap.add_argument(
        "--fp8",
        action="store_true",
        default=False,
        help="Whether to use vortex style FP8. Defaults to False.",
    )
    ap.add_argument(
        "--flash-decode",
        action="store_true",
        default=False,
        help="Whether to use flash decode. Defaults to True.",
    )
    return ap.parse_args()


def infer(
    prompts: list[str],
    output_files: list[Optional[str]],
    seed_offsets: list[int],
    ckpt_dir: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    tensor_parallel_size: int,
    pipeline_model_parallel_size: int,
    context_parallel_size: int,
    ckpt_format: CheckpointFormats = "torch_dist",
    seed: Optional[int] = None,
    num_runs: int = 1,
    vortex_style_fp8: bool = False,
    flash_decode: bool = False,
    return_log_probs: bool = False,
) -> list[InferenceRequest]:
    """Inference workflow for Evo2.

    Args:
        prompts (list[str]): One or more prompts; model and tokenizer are loaded once.
        output_files (list[Optional[str]]): Output JSONL path per prompt, or None to log only.
        seed_offsets (list[int]): Per-prompt seed offsets. With multiple prompts and ``seed`` set, run ``j``
            (1-based, ``j`` in ``1..num_runs``) uses ``seed + seed_offsets[i] + j``. Single-prompt mode uses
            ``seed`` unchanged for every run.
        ckpt_dir (str): Path to checkpoint directory containing pre-trained Evo2 model.
        temperature (float): Temperature during sampling for generation.
        top_k (int): Top K during sampling for generation.
        top_p (float): Top P during sampling for generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        tensor_parallel_size (int): Order of tensor parallelism.
        pipeline_model_parallel_size (int): Order of pipeline parallelism.
        context_parallel_size (int): Order of context parallelism.
        ckpt_format (CheckpointFormats): Checkpoint format to use.
        num_runs (int): Number of generations per prompt (model stays loaded).
        seed (int): Random seed for generation.
        vortex_style_fp8 (bool): Whether to use vortex style FP8.
        flash_decode (bool): Whether to use flash decode.
        return_log_probs (bool): Whether to return log probabilities.

    Returns:
        list[InferenceRequest]: All inference results in order.
    """
    if len(prompts) != len(output_files) or len(prompts) != len(seed_offsets):
        raise ValueError(
            f"prompts ({len(prompts)}), output_files ({len(output_files)}), and seed_offsets "
            f"({len(seed_offsets)}) must have the same length."
        )
    multi_prompt = len(prompts) > 1
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    if model_parallel_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested model parallel size {model_parallel_size} is greater than the "
            f"number of available CUDA devices {torch.cuda.device_count()}"
        )
    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=model_parallel_size,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format=ckpt_format,
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )
    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=10240,  # TODO
        inference_max_seq_length=10240,  # TODO
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
        vortex_style_fp8=vortex_style_fp8,
        flash_decode=flash_decode,
        enable_flash_decode=flash_decode,
    )

    all_results: list[InferenceRequest] = []
    for prompt_idx, prompt in enumerate(prompts):
        output_file = output_files[prompt_idx]
        offset = seed_offsets[prompt_idx]
        for run_idx in range(num_runs):
            if multi_prompt:
                run_seed = seed + offset + (run_idx + 1) if seed is not None else None
            else:
                run_seed = seed if seed is not None else None
            t0 = time.perf_counter_ns()
            # TODO: fix return type in NeMo inference.generate (it is a list[InferenceRequest] not a dict)
            results: list[InferenceRequest] = inference.generate(
                model=inference_wrapped_model,
                max_batch_size=1,  # vortex only supports batch size 1
                tokenizer=mcore_tokenizer,
                prompts=[prompt],
                random_seed=run_seed,
                inference_params=CommonInferenceParams(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    return_log_probs=return_log_probs,
                    num_tokens_to_generate=max_new_tokens,
                ),
            )
            dt = (time.perf_counter_ns() - t0) / 1e9  # seconds
            tokens_per_sec = (len(results[0].generated_text) + 1) / dt  # +1 for the prompt

            label = f"prompt {prompt_idx + 1}/{len(prompts)}" if multi_prompt else "run"
            print(
                f"{label} run {run_idx + 1}/{num_runs}: {dt:.2f}s, {tokens_per_sec:.1f} tokens/sec",
                file=sys.stderr,
            )
            if torch.distributed.get_rank() == 0:
                if output_file is None:
                    logging.info(results)
                else:
                    import json

                    # jsonl, one entry per line
                    data = {
                        "request_id": results[0].request_id,
                        "run": run_idx + 1,
                        "prompt": results[0].prompt,
                        "generated_text": results[0].generated_text,
                        "status": results[0].status.name if results[0].status else None,
                        "num_tokens_to_generate": (
                            results[0].sampling_params.num_tokens_to_generate
                            if results[0].sampling_params
                            else None
                        ),
                    }
                    with open(output_file, "a") as f:
                        json.dump(data, f)
                        f.write("\n")
            all_results.extend(results)

    return all_results


def _read_nonempty_lines(path: str) -> list[str]:
    """Read stripped non-empty lines from a text file."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """Main function for Evo2 inference."""
    args = parse_args()

    if args.prompts_file is not None:
        if not args.output_files_file:
            raise ValueError("--prompts-file requires --output-files-file.")
        prompts = _read_nonempty_lines(args.prompts_file)
        output_files = _read_nonempty_lines(args.output_files_file)
        if len(prompts) != len(output_files):
            raise ValueError(
                f"--prompts-file ({len(prompts)} lines) and --output-files-file ({len(output_files)} lines) "
                "must have the same number of non-empty lines."
            )
        if args.seed_offsets is None:
            seed_offsets = [0] * len(prompts)
        else:
            parts = [p.strip() for p in args.seed_offsets.split(",") if p.strip()]
            seed_offsets = [int(p) for p in parts]
            if len(seed_offsets) != len(prompts):
                raise ValueError(
                    f"--seed-offsets must have {len(prompts)} comma-separated integers (one per prompt), "
                    f"got {len(seed_offsets)}."
                )
    else:
        prompts = [args.prompt]
        output_files = [args.output_file]
        seed_offsets = [0]

    infer(
        prompts=prompts,
        output_files=output_files,
        seed_offsets=seed_offsets,
        ckpt_dir=args.ckpt_dir,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        ckpt_format=args.ckpt_format,
        seed=args.seed,
        num_runs=args.num_runs,
        vortex_style_fp8=args.fp8,  # Vortex only applied FP8 to some layers.
        flash_decode=args.flash_decode,
    )


if __name__ == "__main__":
    main()
