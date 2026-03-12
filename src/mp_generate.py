"""
Multi-GPU generation worker using torch.multiprocessing.

This module is imported by notebooks to enable true parallel generation
across multiple GPUs. Each spawned process loads its own model replica
and generates solutions independently, bypassing the Python GIL.

Usage (from notebook):
    from src.mp_generate import parallel_generate
    results = parallel_generate(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        problems=problems_list,
        num_gpus=8,
        K=4,
        temperature=0.7,
        lora_path=None,  # or path to LoRA checkpoint after fine-tuning
    )
"""

import json
import os
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

# System prompt — must match the notebook's SYSTEM_PROMPT exactly.
# This tells the instruct model to produce "#### <number>" format so
# extract_answer() can parse the output.
SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "showing your reasoning. End your solution with #### followed by "
    "the final numeric answer (e.g., #### 42)."
)


def _format_prompt(question, tokenizer):
    """Format a question using the model's chat template for generation.

    Must match the notebook's format_prompt() exactly — same system prompt,
    same message structure. If these diverge, multi-GPU generation produces
    solutions in a different format than what training expects.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _worker(
    rank,
    world_size,
    model_name,
    problem_chunks,
    result_dir,
    K,
    temperature,
    max_tokens,
    lora_path,
):
    """
    Worker process: loads model on GPU, generates solutions, saves results.

    Each worker is a fully independent process with its own Python interpreter,
    its own GIL, and exclusive access to one GPU. This achieves true parallelism
    that threading cannot provide (because model.generate() holds the GIL during
    the autoregressive decoding loop between CUDA kernel launches).

    Args:
        rank: GPU index (0..world_size-1)
        world_size: total number of GPUs
        model_name: HuggingFace model ID
        problem_chunks: list of (global_idx, question_text) tuples for this worker
        result_dir: directory to write results JSON
        K: number of solutions per problem
        temperature: sampling temperature
        max_tokens: max new tokens per generation
        lora_path: optional path to LoRA adapter weights
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # --- Load model on this GPU ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": rank},
        dtype=torch.float16,
        trust_remote_code=True,
    )

    # Apply LoRA weights if provided (for iterations after fine-tuning)
    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(model, lora_path, device_map={"": rank})

    model.eval()

    # --- Generate solutions ---
    results = {}
    for i, (global_idx, question) in enumerate(problem_chunks):
        prompt = _format_prompt(question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        solutions = []
        for _ in range(K):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # Strip prompt using token length (not character length)
            text = tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True,
            )
            solutions.append(text)

        results[str(global_idx)] = solutions

        if (i + 1) % 10 == 0 or (i + 1) == len(problem_chunks):
            print(f"  [GPU {rank}] {i + 1}/{len(problem_chunks)} problems done")

    # --- Save results to disk (JSON) ---
    out_path = os.path.join(result_dir, f"gpu_{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f)

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


def parallel_generate(
    model_name,
    problems,
    num_gpus,
    K=4,
    temperature=0.7,
    max_tokens=512,
    lora_path=None,
):
    """
    Generate K solutions per problem using torch.multiprocessing across GPUs.

    This is the main entry point called from notebooks. It:
    1. Splits problems into chunks (one per GPU)
    2. Spawns one process per GPU via mp.spawn
    3. Each process loads its own model and generates independently
    4. Results are collected via filesystem (JSON files in a temp dir)

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
        problems: list of dicts with 'question' key
        num_gpus: number of GPUs to use
        K: number of solutions per problem
        temperature: sampling temperature
        max_tokens: max new tokens per generation
        lora_path: optional path to LoRA adapter (applied after base model load)

    Returns:
        dict: {problem_idx (int): [solution_1, ..., solution_K]}
    """
    # --- Distribute problems round-robin ---
    chunks = [[] for _ in range(num_gpus)]
    for idx, problem in enumerate(problems):
        gpu_id = idx % num_gpus
        chunks[gpu_id].append((idx, problem["question"]))

    per_gpu = [len(c) for c in chunks]
    print(
        f"  Distributing {len(problems)} problems across {num_gpus} GPUs "
        f"({', '.join(map(str, per_gpu))} each)"
    )

    # --- Create temp dir for results ---
    result_dir = tempfile.mkdtemp(prefix="mp_gen_")

    # --- Spawn workers ---
    # mp.start_method must be 'spawn' for CUDA (set once globally)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    processes = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(
            target=_worker,
            args=(
                gpu_id,
                num_gpus,
                model_name,
                chunks[gpu_id],
                result_dir,
                K,
                temperature,
                max_tokens,
                lora_path,
            ),
        )
        p.start()
        processes.append((gpu_id, p))

    # --- Wait for all workers ---
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            print(f"  WARNING: GPU {gpu_id} worker exited with code {p.exitcode}")

    # --- Collect results ---
    all_results = {}
    for gpu_id in range(num_gpus):
        result_path = os.path.join(result_dir, f"gpu_{gpu_id}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            for str_idx, solutions in data.items():
                all_results[int(str_idx)] = solutions
            os.remove(result_path)

    # Cleanup temp dir
    try:
        os.rmdir(result_dir)
    except OSError:
        pass

    print(f"  ✓ Collected results for {len(all_results)} problems")
    return all_results
