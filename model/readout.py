"""EXP174 joint physical and independent Bernoulli reachability readouts."""
import torch
from torch import nn

class CoordinateAttentionHead(nn.Module):
    """Two learned axis queries; raw H K/V; one attention and independent MLPs."""
    def __init__(self, dim=256, heads=8, hidden=170, coordinate_limit=63):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.axes = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2*coordinate_limit+2))
            for _ in range(2)
        ])
        for axis in self.axes:
            for module in axis:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=.02)
                    nn.init.zeros_(module.bias)

    def forward(self, state):
        queries = self.queries.expand(len(state), -1, -1)
        read, _ = self.attention(queries, state, state, need_weights=False)
        logits = torch.stack([head(read[:, i]) for i, head in enumerate(self.axes)], dim=1).float()
        return torch.cat((torch.full_like(logits[..., :1], -torch.inf), logits[..., 1:]), dim=-1)

class JointReadout(nn.Module):
    """P(C,X,Y|H)=P(C|H)P(X|H,C)P(Y|H,C); C=1 means collision."""
    def __init__(self, coordinate):
        super().__init__()
        dim = coordinate.queries.shape[-1]
        extra=(torch.randn(1,1,dim)*.02).to(coordinate.queries.device)
        self.queries = nn.Parameter(torch.cat((coordinate.queries.detach().clone(),extra),1))
        self.attention = coordinate.attention
        self.axes = coordinate.axes  # SAME axis MLPs, shared across both conditions.
        self.conditions = nn.Parameter(torch.randn(2,dim)*.02)
        self.event = nn.Linear(dim,2)

    def forward(self, state):
        r = self.attention(self.queries.expand(len(state),-1,-1),state,state,need_weights=False)[0]
        event = self.event(r[:,2]).float()
        conditional = torch.stack([head(r[:,None,j]+self.conditions[None]) for j,head in enumerate(self.axes)],2).float()
        conditional = torch.cat((torch.full_like(conditional[...,:1],-torch.inf),conditional[...,1:]),-1)
        log_c = event.log_softmax(-1)
        log_xy = conditional[...,1:].log_softmax(-1)
        marginal = torch.logsumexp(log_c[:,:,None,None]+log_xy,1)
        marginal = torch.cat((torch.full_like(marginal[...,:1],-torch.inf),marginal),-1)
        return dict(event=event,collision_logit=event[:,1]-event[:,0],conditional=conditional,
                    log_c=log_c,log_xy=log_xy,marginal=marginal)

    @staticmethod
    def decode(joint, position, displacement):
        """Condition joint on the two legal outcomes, no action is forbidden."""
        candidates = torch.stack((position+displacement,position),1).long()
        limit = (joint['log_xy'].shape[-1]-1)//2
        valid = ((candidates>=-limit)&(candidates<=limit)).all(-1)
        if not bool(valid[:,1].all()): raise ValueError('Current branch position out of coordinate range')
        index = (candidates+limit).clamp(0,2*limit).unsqueeze(-1)
        scores = joint['log_c']+joint['log_xy'].gather(-1,index).squeeze(-1).sum(-1)
        scores = scores.masked_fill(~valid,-torch.inf)
        probability = scores.softmax(-1)
        collision = scores.argmax(-1).bool()
        next_position = candidates[torch.arange(len(position),device=position.device),collision.long()]
        return dict(probability=probability,collision=collision,position=next_position,scores=scores)

class ActionReadout(nn.Module):
    def __init__(self, dim=256, heads=8, hidden=170, action_count=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1,1,dim)*.02)
        self.attention = nn.MultiheadAttention(dim,heads,dropout=0.,batch_first=True)
        self.output = nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,action_count))

    def forward(self,state,return_logits=False):
        r = self.attention(self.query.expand(len(state),-1,-1),state,state,need_weights=False)[0][:,0]
        logits=self.output(r).float()
        return logits if return_logits else logits.sigmoid()
