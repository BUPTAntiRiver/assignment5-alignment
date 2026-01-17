"""
A script to evaluate Qwen 2.5 Maht 1.5B model zero-shot performance on given dataset.
"""

import os
from vllm import LLM, SamplingParams
from utils import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn

if __name__ == "__main__":
    model_path = "data/models/Qwen2.5-Math-1.5B"
    vllm_model = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop="</answer>", include_stop_str_in_output=True)
    prompts = []
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompts.append(f.read())

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=sampling_params,
    )
