"""
File: torch_functions.py
Author: Matthew Allen

Description:
    A helper file for misc. PyTorch functions.

"""


import torch.nn as nn
import torch
from typing import Tuple
import numpy as np
from numba import njit

@njit
def _gae_returns_loop(delta: np.ndarray, gae_factor: np.ndarray, rews: np.ndarray, ret_factor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    length = len(delta)
    advantages = np.zeros_like(delta)
    returns = np.zeros_like(delta)
    
    last_gae = 0.0
    last_ret = 0.0
    
    for t in range(length - 1, -1, -1):
        last_gae = delta[t] + gae_factor[t] * last_gae
        last_ret = rews[t] + ret_factor[t] * last_ret
        advantages[t] = last_gae
        returns[t] = last_ret
        
    return advantages, returns


class MapContinuousToAction(nn.Module):
    """
    A class for policies using the continuous action space. Continuous policies output N*2 values for N actions where
    each value is in the range [-1, 1]. Half of these values will be used as the mean of a multi-variate normal distribution
    and the other half will be used as the diagonal of the covariance matrix for that distribution. Since variance must
    be positive, this class will map the range [-1, 1] for those values to the desired range (defaults to [0.1, 1]) using
    a simple linear transform.
    """
    def __init__(self, range_min=0.1, range_max=1):
        super().__init__()

        tanh_range = [-1, 1]
        self.m = (range_max - range_min) / (tanh_range[1] - tanh_range[0])
        self.b = range_min - tanh_range[0] * self.m

    def forward(self, x):
        n = x.shape[-1] // 2
        # map the right half of x from [-1, 1] to [range_min, range_max].
        return x[..., :n], x[..., n:] * self.m + self.b


def compute_gae(rews, dones, truncated, values, gamma=0.99, lmbda=0.95, return_std=1):
    """
    Function to estimate the advantage function for a series of states and actions using the
    general advantage estimator (GAE).
    """

    # Ensure all inputs are PyTorch tensors on the same device
    # If they are numpy arrays or lists, convert them
    # We assume 'values' is already a tensor (from previous steps).
    device = values.device

    # Helper to convert to tensor if needed and ensure 1D shape
    def to_tensor(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=device, dtype=torch.float32)
        return x.view(-1)

    rews = to_tensor(rews)
    dones = to_tensor(dones)
    truncated = to_tensor(truncated)
    
    # Values might be (N+1) length, keep as is
    # but ensure tensor
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, device=device, dtype=torch.float32)
    values = values.view(-1)

    next_values = values[1:]
    current_values = values[:-1]

    # Calculate delta for all steps at once via vectorization
    # delta = r + gamma * V(s') * (1-done) - V(s)
    
    # Apply standard normalization clippping to rewards if requested
    if return_std is not None:
        # Clone rews to avoid modifying original
        norm_rews = torch.clamp(rews / return_std, min=-10.0, max=10.0)
    else:
        norm_rews = rews

    # 1 - dones
    not_dones = 1.0 - dones
    not_truncs = 1.0 - truncated

    # delta = r + gamma * next_val * not_done - val
    delta = norm_rews + gamma * next_values * not_dones - current_values

    # Now we compute the advantages recursively.
    # A_t = delta_t + (gamma * lambda) * not_done * not_trunc * A_{t+1}
    # This loop must be reversed.
    
    gae = 0.0
    advantages = torch.zeros_like(rews)
    
    # We loop backwards
    # It is faster to use a python loop over tensors than converting to list
    # But even better would be JIT, but for now just staying on GPU is enough speedup 
    # compared to CPU list iteration.
    
    # Convert scalar constants to tensors or basic floats for the loop
    factor = gamma * lmbda

    # We use a simple loop. Since everything is on GPU, element access might be slow
    # unless we use torch.jit.script. 
    # Let's try to stick to basic vector ops, but GAE is recursive.
    # So we loop.
    
    # Move to CPU for JIT loop to avoid CUDA launch overhead per element
    delta_cpu = delta.cpu().numpy()
    gae_factor_cpu = (not_dones * not_truncs * factor).cpu().numpy()
    
    # For Returns (MC) calculation
    rews_cpu = rews.cpu().numpy()
    ret_factor_cpu = (not_dones * not_truncs * gamma).cpu().numpy()
    
    # Use JIT compiled loop on CPU tensors
    advantages_cpu, mc_returns_cpu = _gae_returns_loop(delta_cpu, gae_factor_cpu, rews_cpu, ret_factor_cpu)
    
    advantages = torch.as_tensor(advantages_cpu, device=device, dtype=torch.float32)
    mc_returns = torch.as_tensor(mc_returns_cpu, device=device, dtype=torch.float32)

    # Lambda Returns = Advantage + Value
    lambda_returns = advantages + current_values

    return lambda_returns, advantages, mc_returns


class MultiDiscreteRolv(nn.Module):
    """
    A class to handle the multi-discrete action space in Rocket League. There are 8 potential actions, 5 of which can be
    any of {-1, 0, 1} and 3 of which can be either of {0, 1}. This class takes 21 logits, appends -inf to the final 3
    such that each of the 8 actions has 3 options (to avoid a ragged list), then builds a categorical distribution over
    each class for each action. Credit to Rolv Arild for coming up with this method.
    """
    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins

    def make_distribution(self, logits):
        """
        Function to make the multi-discrete categorical distribution for a group of logits.
        :param logits: Logits which parameterize the distribution.
        :return: None.
        """

        # Split the 21 logits into the expected bins.
        logits = torch.split(logits, self.bins, dim=-1)

        # Separate triplets from the split logits.
        triplets = torch.stack(logits[:5], dim=-1)

        # Separate duets and pad the final dimension with -inf to create triplets.
        duets = torch.nn.functional.pad(torch.stack(logits[5:], dim=-1), pad=(0,0,0,1), value=float("-inf"))

        # Un-split the logits now that the duets have been converted into triplets and reshape them into the correct shape.
        logits = torch.cat((triplets, duets), dim=-1).swapdims(-1, -2)

        # Construct a distribution with our fixed logits.
        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) # Unsure about this sum operation.
