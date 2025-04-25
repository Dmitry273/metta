from typing import List, Union

import torch


def normalize_logits(logits):
    """
    Normalize logits by subtracting the logsumexp.

    Args:
        logits: Tensor of unnormalized logits

    Returns:
        Normalized logits (log probabilities)
    """
    return logits - logits.logsumexp(dim=-1, keepdim=True)


def log_prob(normalized_logit, value):
    """
    Compute log probability of a value given normalized logits.

    Args:
        normalized_logit: Normalized log probabilities
        value: The value to compute probability for

    Returns:
        Log probability of the value
    """
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, normalized_logit)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(normalized_logit):
    """
    Compute entropy of a categorical distribution given normalized logits.

    Args:
        normalized_logit: Normalized log probabilities

    Returns:
        Entropy of the distribution
    """
    min_real = torch.finfo(normalized_logit.dtype).min
    normalized_logit = torch.clamp(normalized_logit, min=min_real)
    probs = torch.exp(normalized_logit)
    return -(probs * normalized_logit).sum(-1)


def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]], action=None, verbose=False):
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities, either a single tensor or a list of tensors
        action: Optional pre-specified actions to compute probabilities for
        verbose: If True, print debug information

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
    """
    # Check if logits is a single tensor and convert to list for consistent handling
    is_single_tensor = isinstance(logits, torch.Tensor)
    if is_single_tensor:
        logits = [logits]

    # Normalize each logits tensor
    normalized_logits = [normalize_logits(logit) for logit in logits]

    batch_size = normalized_logits[0].shape[0]
    num_logits = len(normalized_logits)

    if action is None:
        # Pre-allocate action tensor with correct shape
        device = normalized_logits[0].device
        action = torch.empty((batch_size, num_logits), dtype=torch.long, device=device)

        # Debug output if verbose mode is enabled
        if verbose:
            print(f"logits has len {num_logits}")
            lgt = logits[0]
            print(f"logits[0] {lgt}, has shape {lgt.shape}")
            lgt_prob = torch.exp(normalized_logits[0])
            print(f"probabilities: {lgt_prob}")

        # Sample actions directly into pre-allocated tensor
        for i, norm_logit in enumerate(normalized_logits):
            probs = torch.exp(norm_logit)
            action[:, i] = torch.multinomial(probs, 1).flatten()

        if verbose:
            print(f"action has shape {action.shape}")
    else:
        # Fast reshape of provided action if needed
        if action.dim() == 1:
            if len(logits) == 1:
                # Single logit case
                action = action.view(batch_size, 1)
            else:
                # Multiple logits case
                action = action.view(batch_size, -1)

    # Pre-allocate tensors for log probabilities and entropy
    logprob = torch.zeros(batch_size, device=normalized_logits[0].device)
    logits_entropy = torch.zeros(batch_size, device=normalized_logits[0].device)

    # Compute log probabilities and entropy
    for i, norm_logit in enumerate(normalized_logits):
        # Extract actions for this logit
        act_i = action[:, i].unsqueeze(-1) if action.dim() > 1 else action.unsqueeze(-1)

        # Get log probabilities and add them to the total
        logprob_i = log_prob(norm_logit, act_i if act_i.dim() > 1 else act_i.squeeze(-1))
        logprob.add_(logprob_i)

        # Get entropy and add it to the total
        entropy_i = entropy(norm_logit)
        logits_entropy.add_(entropy_i)

    # Format return values based on whether it's a single tensor or a list
    if is_single_tensor and action.dim() > 1:
        action = action.squeeze(1)

    return action, logprob, logits_entropy, normalized_logits
