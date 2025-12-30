"""
Custom callbacks for PyTorch Lightning training.
"""
import torch
from lightning.pytorch.callbacks import Callback


class GradNormCallback(Callback):
    """
    Callback to log the gradient norm, weight norm, and update ratio.
    
    Uses on_before_optimizer_step hook to capture gradient norm after
    gradient clipping (if enabled) but before optimizer step.
    """

    def on_before_optimizer_step(self, trainer, model, optimizer):
        g_norm = gradient_norm(model)
        w_norm = weight_norm(model)
        
        # Get learning rate from the first param group
        lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
        
        # Avoid division by zero
        ratio = (lr * g_norm) / (w_norm + 1e-6)

        model.log("grad_norm", g_norm)
        model.log("weight_norm", w_norm)
        model.log("update_ratio", ratio)


def weight_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the weight norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Weight norm.
    """
    params = [p for p in model.parameters() if p is not None]
    if not params:
        return torch.tensor(0.0)
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type) for p in params]), norm_type)
    return total_norm


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Gradient norm.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm
