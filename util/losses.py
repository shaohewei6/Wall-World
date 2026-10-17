"""Original EXP174 objectives; only executed actions receive reachability labels."""
import math
import torch
from torch.nn import functional as F
if "." in __package__:
    from ..model.coordinates import coordinate_targets
else:
    from model.coordinates import coordinate_targets

def gaussian_targets(coordinates, sigma=1., limit=63):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be positive and finite")
    coordinate_targets(coordinates, limit)  # Validate before float conversion.
    classes = torch.arange(-limit, limit+1, device=coordinates.device, dtype=torch.float32)
    return (-.5 * ((classes - coordinates.float().unsqueeze(-1)) / sigma).square()).softmax(-1)

def action_items(logits, action, collision, back):
    """Only the executed direction receives a binary label; reverse is diagnostic only."""
    chosen_logits=logits.gather(1,action[:,None]).squeeze(1)
    target=(~collision).float()
    forward=F.softplus(-chosen_logits)*(~collision)
    hit=F.softplus(chosen_logits)*collision
    loss=F.binary_cross_entropy_with_logits(chosen_logits,target,reduction='none')*(1+3*collision.float())
    probability=logits.sigmoid()
    reverse=probability.gather(1,back.clamp_min(0)[:,None]).squeeze(1)
    return loss,dict(chosen=chosen_logits.sigmoid(),back=reverse,success=forward,hit=hit,backward=torch.zeros_like(loss))

def js(p,q):
    """Mean JS divergence of the eight Bernoulli marginals, not categorical JS."""
    p=p.float().clamp(1e-7,1-1e-7);q=q.float().clamp(1e-7,1-1e-7);m=(p+q)/2
    def kl(x):return x*(x.log()-m.log())+(1-x)*(torch.log1p(-x)-torch.log1p(-m))
    return .5*(kl(p)+kl(q)).mean(-1)
