import torch
from typing import Callable

def compute_group_normalized_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        advantage_eps: float,
        normalize_by_std: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Return:
        advantages, raw_rewards, metadata
    """
    rewards = [reward_fn(rr, rgt)["reward"] for rr, rgt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    num_groups = len(rollout_responses) // group_size
    advantages = torch.zeros_like(raw_rewards)
    for i in range(num_groups):
        group_rewards = raw_rewards[i * group_size: (i + 1) * group_size]
        mean_reward = torch.mean(group_rewards)
        if normalize_by_std:
            std_reward = torch.std(group_rewards)
            std_reward += advantage_eps
            group_advantages = (group_rewards - mean_reward) / std_reward
        else:
            group_advantages = group_rewards - mean_reward
        advantages[i * group_size: (i + 1) * group_size] = group_advantages
    
    metadata = {
        "mean_raw_reward": torch.mean(raw_rewards).item(),
        "std_raw_reward": torch.std(raw_rewards).item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs