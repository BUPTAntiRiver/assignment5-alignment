import torch
from einops import rearrange
from typing import Callable, Literal

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


def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_range: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Return:
        loss, metadata
    """
    weights = torch.exp(policy_log_probs - old_log_probs)
    clipped_weights = torch.clamp(weights, 1.0 - clip_range, 1.0 + clip_range)
    clipped_loss = clipped_weights * advantages
    unclipped_loss = weights * advantages
    loss = -torch.min(clipped_loss, unclipped_loss)
    metadata = {}
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            clip_range=cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss, metadata


def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None
) -> torch.Tensor:
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim, keepdim=False)
    count = mask.sum(dim=dim, keepdim=False)
    return masked_sum / count