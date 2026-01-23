import os
from vllm import LLM, SamplingParams
from typing import Callable
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # load gsm8k test examples
    test_examples = []
    gsm8k_test_path = "data/gsm8k/test.jsonl"
    with open(gsm8k_test_path, "r") as f:
        for line in tqdm(f, desc="Loading test examples"):
            if line.strip():
                test_examples.append(json.loads(line))

    # format question prompts using the r1-zero prompt
    formatted_prompts = []
    for example in tqdm(test_examples, desc="Formatting prompts"):
        question = example["question"]
        formatted_prompt = prompts[0].replace("{question}", question)
        formatted_prompts.append(formatted_prompt)

    # generate model responses
    responses = []
    for response in tqdm(vllm_model.generate(
        formatted_prompts,
        sampling_params=eval_sampling_params,
    ), desc="Generating responses"):
        response_text = response.outputs[0].text
        responses.append(response_text)

    # calculate evaluation metrics
    results = []
    for i in tqdm(range(len(test_examples)), desc="Calculating evaluation metrics"):
        ground_truth = test_examples[i]["answer"]
        model_response = responses[i]
        metrics = reward_fn(ground_truth, model_response)
        results.append({
            "question": test_examples[i]["question"],
            "ground_truth": ground_truth,
            "model_response": model_response,
            "metrics": metrics,
        })

    # serialize results to disk
    os.makedirs("results", exist_ok=True)
    model_name = vllm_model.llm_engine.model_config.model
    results_path = os.path.join("results", f"{model_name}_evaluation_results.jsonl")
    with open(results_path, "w") as f:
        for result in tqdm(results, desc="Saving results"):
            f.write(json.dumps(result) + "\n")


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    assert len(prompt_strs) == len(output_strs)

    prompt_enc = tokenizer(prompt_strs, add_special_tokens=True)
    output_enc = tokenizer(output_strs, add_special_tokens=True)

    # strip EOS from outputs (Qwen adds it anyway)
    output_ids_list = []
    for ids in output_enc["input_ids"]:
        if ids and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
        output_ids_list.append(ids)

    input_ids_list = []
    response_mask_list = []
    

    for p_ids, o_ids in zip(prompt_enc["input_ids"], output_ids_list):
        full_ids = p_ids + o_ids

        input_ids_list.append(torch.tensor(full_ids))

        mask = [0] * (len(p_ids) - 1) + [1] * len(o_ids)
        response_mask_list.append(torch.tensor(mask, dtype=torch.bool))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = input_ids.clone()

    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]

    response_mask = torch.nn.utils.rnn.pad_sequence(
        response_mask_list,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # entropy = -sum(p * log(p))
    # logsumexp = log(sum(exp(logits)))
    # p = exp(logits) / sum(exp(logits)) = exp(logits - logsumexp)
    # entropy = -sum( exp(logits - logsumexp) * (logits - logsumexp) )
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    entropy = -torch.sum(
        torch.exp(logits - lse) * (logits - lse),
        dim=-1,
    )
    return entropy


def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(
        log_softmax,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    result = {
        "log_probs": log_probs,
    }
    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy
    return result