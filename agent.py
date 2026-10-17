"""EXP174 E6 inference and final candidate_01_v1 navigation in a single Agent."""
import base64
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
import zlib
import numpy as np
import torch
from torch import nn

if __package__:
    from .model.encoder import AgentDINOv2Encoder
    from .model.dynamic import Dynamic
    from .model.observation import ConcatObservation
    from .model.normalization import FixedRMSNorm
    from .model.readout import CoordinateAttentionHead, JointReadout, ActionReadout
    from .util.config import model_config, load_navigation, load_agent_config, digest, save_resolved_config, navigation_config as normalize_navigation
    from .util.checkpoint import load_model_checkpoint
else:
    from model.encoder import AgentDINOv2Encoder
    from model.dynamic import Dynamic
    from model.observation import ConcatObservation
    from model.normalization import FixedRMSNorm
    from model.readout import CoordinateAttentionHead, JointReadout, ActionReadout
    from util.config import model_config, load_navigation, load_agent_config, digest, save_resolved_config, navigation_config as normalize_navigation
    from util.checkpoint import load_model_checkpoint


MOVES = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1))
_MOVES = np.asarray(MOVES, dtype=np.int8)

def kernel(p, center, c):
    delta=np.asarray(p,dtype=np.float64)-np.asarray(center,dtype=np.float64)
    rho=c['field']['rho']
    return rho/np.sqrt((delta*delta).sum(-1)+rho*rho)

def weights(records, step, c):
    ages=np.array([step-r['created'] for r in records],np.float64)
    assert (ages>=0).all()
    return np.exp2(-ages/c['field']['half_life'])

def history_evidence(recent, goal, c):
    cfg=c['registration'];p=np.asarray(recent[-cfg['window']:],dtype=np.float64)
    ready=len(p)==cfg['window'];unique=len(set(map(tuple,p)))
    early=late=improvement=None
    if ready:
        d=np.linalg.norm(p-np.asarray(goal),axis=1);half=cfg['half_window']
        early=float(d[:half].min());late=float(d[half:].min());improvement=early-late
    repeat=1-unique/len(p) if len(p) else 0.
    stagnant=bool(ready and improvement<cfg['distance_improvement'] and repeat>=cfg['repeat_fraction'])
    return dict(ready=ready,positions=len(p),unique_positions=unique,repeat_fraction=repeat,
                early_best_distance=early,late_best_distance=late,improvement=improvement,stagnant=stagnant)

class PointTable:
    def __init__(self, goal, config):
        self.goal = tuple(goal)
        self.c = config
        self.records, self.events, self.next_id = [], [], 0
        if config['field']['initial_repulsion']:
            self.register((0, 0), 0, 'initial')

    def register(self,position,step,kind='extremum'):
        position=list(map(int,position))
        interval = self.c['field']['registration_interval']
        if interval and self.events and step-self.events[-1]['step'] < interval:
            return None
        existing=next((r for r in self.records if r['position']==position),None)
        if existing is not None:
            r=existing;r['triggers']+=1;event='duplicate'
        else:
            capacity = self.c['field']['capacity']
            if capacity is not None and len(self.records) >= capacity:
                oldest = min(self.records, key=lambda record: (record['created'], record['id']))
                self.records.remove(oldest)
                self.events.append(dict(event='evict', step=int(step), position=deepcopy(oldest['position']),
                                        point_id=oldest['id'], kind=oldest['kind']))
            r=dict(id=self.next_id,position=position,created=int(step),kind=kind,triggers=1)
            self.next_id+=1;self.records.append(r);event='add'
        ev=dict(event=event,step=int(step),position=position,point_id=r['id'],kind=kind)
        self.events.append(ev);return deepcopy(ev)

    def state(self):
        return deepcopy(dict(records=self.records,events=self.events,next_id=self.next_id))

    def restore(self,s):
        for k,v in deepcopy(s).items():setattr(self,k,v)

def goal_gate(position, goal, config):
    delta = np.asarray(position, dtype=np.float64)-np.asarray(goal, dtype=np.float64)
    squared = (delta*delta).sum(-1)/config['field']['goal_gate_length']**2
    return -np.expm1(-squared)

def local_kernel(position, center, config):
    delta=np.asarray(position,dtype=np.float64)-np.asarray(center,dtype=np.float64)
    ratio=(delta*delta).sum(-1)/config['field']['repulsion_radius']**2
    return np.maximum(1.-ratio,0.)**2

def potential_components(before, after, goal, records, step, config):
    goal_gain = config['field']['goal_strength']*(kernel(after, goal, config)-kernel(before, goal, config))
    repel = np.zeros_like(goal_gain)
    b0, b1 = goal_gate(before, goal, config), goal_gate(after, goal, config)
    for record, weight in zip(records, weights(records, step, config)):
        repel += config['field']['repel_strength']*weight*(
            b0*local_kernel(before, record['position'], config)-b1*local_kernel(after, record['position'], config))
    return goal_gain, repel

def potential(position, goal, records, step, config):
    result = config['field']['goal_strength']*kernel(position, goal, config)
    for record, weight in zip(records, weights(records, step, config)):
        result = result-config['field']['repel_strength']*weight*goal_gate(position, goal, config)*local_kernel(position, record['position'], config)
    return result

def score_bank(bank, table, goal, step, config, guard_hits):
    assert config['gamma'] == 1.
    endpoints = np.take_along_axis(bank['positions'], (bank['lengths']-1)[:, None, None], axis=1)[:, 0]
    goal_gain, repel = potential_components(bank['root'], endpoints, goal, table.records, step, config)
    arrival = (endpoints == np.asarray(goal)).all(-1)
    components = np.stack([arrival*config['reward']['success'], goal_gain, repel], 1)
    safe = (bank['collision_bits'] == 0) & ~np.asarray(guard_hits, dtype=bool)[bank['actions'][:, 0]]
    scores = np.where(safe, components.sum(1), -np.inf)
    assert np.isfinite(components).all() and not np.isnan(scores).any()
    return scores, components, safe

def registration(table, step, scores, safe, position, recent, c, has_safe_goal=False):
    retained=np.asarray(scores,dtype=np.float64)[np.asarray(safe,dtype=bool)]
    assert np.isfinite(retained).all()
    # The second arm logs history only; the first arm retains the EXP177 criterion.
    history=history_evidence(recent,table.goal,c)
    no_gain=bool(len(retained)>0 and np.max(retained)<=c['registration']['epsilon'])
    eligible=bool(len(retained)>0 and not has_safe_goal and ((c['registration']['no_gain_enabled'] and no_gain) or (c['registration']['history_enabled'] and history['stagnant'])))
    event=table.register(position,step) if eligible else None
    return event,dict(eligible=eligible,safe_count=len(retained),
        max_score_before=float(np.max(retained)) if len(retained) else None,
        negative_safe_count=int(np.sum(retained<0.)),zero_safe_count=int(np.sum(retained==0.)),
        positive_safe_count=int(np.sum(retained>0.)),no_gain=no_gain,has_safe_goal=bool(has_safe_goal),
        history=history,reasons=([name for name,flag in [('no_gain',c['registration']['no_gain_enabled'] and no_gain),('stagnation',c['registration']['history_enabled'] and history['stagnant'])] if flag] if eligible else []))

def load_policy():
    return load_navigation()

def draw(p, rng):
    p = p.detach().double().cpu().numpy(); p /= p.sum(-1, keepdims=True)
    return np.minimum((rng.random(len(p))[:, None] > np.cumsum(p, axis=-1)).sum(-1), 7)

def json_scores(value):
    """Strict JSON: explicit strings preserve infinities; NaN is always an error."""
    if isinstance(value, np.ndarray):
        return json_scores(value.tolist())
    if isinstance(value, dict):
        return {k: json_scores(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_scores(v) for v in value]
    if isinstance(value, (float, np.floating)):
        assert not math.isnan(value)
        return ('+inf' if value > 0 else '-inf') if math.isinf(value) else float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value

def probabilities(scores, temperature):
    scores = np.asarray(scores, np.float64)
    assert scores.ndim == 1 and len(scores) and temperature > 0
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    finite = np.isfinite(scores)
    if not finite.any():
        return None
    p = np.zeros_like(scores)
    with np.errstate(under='ignore'):
        p[finite] = np.exp((scores[finite]-scores[finite].max()) / temperature)
    p /= p.sum()
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.)
    return p

def choose(p, rng):
    p = np.asarray(p, np.float64)
    assert np.isfinite(p).all() and np.all(p >= 0) and np.isclose(p.sum(), 1.)
    eligible = np.flatnonzero(p > 0)
    cumulative = np.cumsum(p[eligible]); cumulative[-1] = 1.
    return int(eligible[np.searchsorted(cumulative, rng.random(), side='right')])

def segment_end(ledger, hit, mismatch, offset, length):
    if ledger.position == ledger.goal: return 'success'
    if ledger.remaining == 0: return 'timeout'
    if hit: return 'collision'
    if mismatch: return 'prediction_mismatch'
    if offset == length: return 'completed'
    return None

class Ledger:
    def __init__(self,goal,limit,c):
        self.goal=tuple(map(int,goal));self.limit=int(limit);self.c=c;self.position=(0,0);self.steps=0;self.collisions=0
        self.visited={(0,0)};self.edges={};self.ended=False
    @property
    def remaining(self):return self.limit-self.steps
    def clone(self):
        other=Ledger(self.goal,self.limit,self.c);other.position=self.position;other.steps=self.steps;other.collisions=self.collisions
        other.visited=self.visited.copy();other.edges=self.edges.copy();other.ended=self.ended;return other
    def append(self,action,position,hit):
        assert not self.ended and self.remaining>0
        a=int(action);p=tuple(map(int,position));hit=bool(hit);before=self.position
        assert p==tuple(np.asarray(before)+_MOVES[a]*(not hit))
        d0=math.hypot(self.goal[0]-before[0],self.goal[1]-before[1]);d1=math.hypot(self.goal[0]-p[0],self.goal[1]-p[1])
        first=p not in self.visited;m=0 if hit else self.edges.get((before,a),0);n=self.collisions+int(hit);rc=self.c['reward']
        components=np.array([rc['success']*int(p==self.goal),rc['progress']*(d0-d1),rc['first_visit']*int(first),
            -rc['collision_first']*math.pow(rc['collision_base'],n-1) if hit else 0.,-rc['repeat_coefficient']*math.log1p(m)/math.log(2)],dtype=np.float64)
        r=float(components.sum());assert math.isfinite(r)
        if not hit:self.edges[(before,a)]=m+1
        self.visited.add(p);self.position=p;self.steps+=1;self.collisions=n;self.ended=p==self.goal or self.remaining==0
        return dict(reward=r,components=components.tolist(),success=p==self.goal,terminated=self.ended,collision=hit,first_visit=first,repeat_previous=m,collision_count=n)
    def state(self):return dict(goal=self.goal,limit=self.limit,position=self.position,steps=self.steps,collisions=self.collisions,visited=sorted(self.visited),edges=[(p,a,n) for (p,a),n in sorted(self.edges.items())],ended=self.ended)
    @classmethod
    def restore(cls,s,c):
        x=cls(s['goal'],s['limit'],c);x.position=tuple(s['position']);x.steps=s['steps'];x.collisions=s['collisions'];x.visited=set(map(tuple,s['visited']));x.edges={(tuple(p),int(a)):int(n) for p,a,n in s['edges']};x.ended=s['ended']
        assert x.steps<=x.limit and x.collisions<=x.steps and x.position in x.visited;return x

def proposal_probabilities(first_actions, depth_index, c):
    """Independent angle proposal; depth is zero-based and first_actions never drifts."""
    first=np.asarray(first_actions,dtype=np.int64).reshape(-1)
    assert 0<=depth_index<c['horizon'] and ((first>=0)&(first<8)).all()
    if depth_index==0:return np.full((len(first),8),1/8,dtype=np.float64)
    sigma=c['sigma_degrees'][depth_index]
    assert sigma is not None and sigma>0
    motion=_MOVES.astype(np.float64)
    angles=np.degrees(np.arctan2(motion[:,1],motion[:,0]))
    delta=(angles[None,:]-angles[first,None]+180.)%360.-180.
    weights=np.exp(-.5*(delta/sigma)**2)
    return weights/weights.sum(-1,keepdims=True)

def proposal(base, scores, epsilon=.1):
    """90% normalized(base * reachability) + 10% original angular proposal."""
    base = np.asarray(base, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    assert base.shape == scores.shape and base.shape[-1] == 8
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    assert np.isfinite(base).all() and (base >= 0).all()
    base = base / base.sum(-1, keepdims=True)
    joint = base * scores
    weighted = np.divide(joint, joint.sum(-1, keepdims=True), out=base.copy(),
                         where=joint.sum(-1, keepdims=True) > 1e-30)
    result = (1-epsilon)*weighted + epsilon*base
    return result / result.sum(-1, keepdims=True)

def pack_scores(array):
    values = np.ascontiguousarray(array, dtype='<f4')
    return dict(shape=list(values.shape), dtype='<f4',
                zlib_base64=base64.b64encode(zlib.compress(values.tobytes(), 6)).decode())

def unpack_scores(value):
    raw = zlib.decompress(base64.b64decode(value['zlib_base64']))
    return np.frombuffer(raw, dtype=value['dtype']).reshape(value['shape'])

class Agent(nn.Module):
    def __init__(self, config=None, *, device="cpu", precision="float32", navigation=None):
        super().__init__()
        self.config = model_config(config)
        c = self.config
        if precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if precision == "bfloat16" and torch.device(device).type != "cuda":
            raise ValueError("Reference bfloat16 inference requires CUDA")
        self.precision = precision
        self.inner_token_count, self.inner_state_dim, self.action_count = c['inner_token_count'], c['inner_state_dim'], c['action_count']
        self.coordinate_limit = c['coordinate_limit']
        self.navigation_config = load_policy() if navigation is None else normalize_navigation(navigation)
        self.resolved_config = None
        self.coordinate_sigma = float(c["coordinate_soft_sigma"])
        self.encoder = AgentDINOv2Encoder(c["lora_config"], image_size=c['image_size'], patch_size=c['encoder_patch_size'],
            pretrain_image_size=c['encoder_pretrain_image_size'], layer_scale=c['encoder_layer_scale'],
            interpolate_antialias=c['encoder_interpolate_antialias'], interpolate_offset=c['encoder_interpolate_offset'])
        dim, tokens, visual, visual_tokens = self.inner_state_dim, self.inner_token_count, c['visual_state_dim'], c['visual_token_count']
        self.h_identity = nn.Embedding(tokens, dim)
        nn.init.normal_(self.h_identity.weight, std=0.02)
        self.dynamic = Dynamic(inner_state_dim=dim, action_count=self.action_count, depth=c["dynamic_depth"],
                               heads=c["dynamic_heads"], mlp_ratio=c["dynamic_mlp_ratio"],
                               dropout=c["dynamic_dropout"], inner_token_count=tokens,
                               identity=self.h_identity, state_rms_limit=c["state_target_rms"])
        self.observation = ConcatObservation(
            visual_state_dim=visual, inner_state_dim=dim, visual_token_count=visual_tokens, inner_token_count=tokens,
            visual_depth=c["observation_depth"], heads=c["observation_heads"],
            mlp_ratio=c["observation_mlp_ratio"], dropout=c["observation_dropout"],
            identity=self.h_identity, state_rms_limit=c["state_target_rms"],
            initial_gate=c["observation_initial_gate"], coordinate_limit=self.coordinate_limit)
        for module in (self.dynamic, self.observation):
            module.state_norm = FixedRMSNorm(c["state_target_rms"])
        for name in ("visual", "coordinate"):
            getattr(self.observation, name + "_correction").correction_norm = FixedRMSNorm(c["state_target_rms"])
        # Keep historical checkpoint names, including the visual readout's root attributes.
        self.visual_queries = nn.Parameter(torch.zeros(1, visual_tokens, dim))
        self.visual_query_norm = nn.LayerNorm(dim)
        self.visual_source_norm = nn.LayerNorm(dim)
        self.visual_cross_attention = nn.MultiheadAttention(dim, c["visual_predictor_heads"],
                                                          dropout=c["visual_predictor_dropout"], batch_first=True)
        self.visual_output_norm = nn.LayerNorm(dim)
        self.visual_output_projection = nn.Linear(dim, visual)
        self.joint_readout = JointReadout(CoordinateAttentionHead(dim, c["coordinate_attention_heads"], c["coordinate_hidden_dim"], self.coordinate_limit))
        self.action_readout = ActionReadout(dim, c['action_attention_heads'], c['action_hidden_dim'], self.action_count)
        self.register_buffer("h_initial", torch.zeros(1, tokens, dim))
        self.register_buffer("h", None, persistent=False)
        self.register_buffer("relative_position", None, persistent=False)
        self.to(device)
        self.eval().requires_grad_(False)
        self.reference = None
        self._navigation = None

    @property
    def device(self):
        return self.h_initial.device

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.bfloat16)
                if self.precision == "bfloat16" else nullcontext())

    @classmethod
    def from_checkpoint(cls, checkpoint, *, encoder_reference=None, device="cpu", precision="float32", config=None,
                        checkpoint_sha256=None, encoder_sha256=None, navigation=None):
        state, reference = load_model_checkpoint(checkpoint, encoder_reference,
            expected_sha256=checkpoint_sha256, encoder_sha256=encoder_sha256)
        model = cls(config, device=device, precision=precision, navigation=navigation)
        recorded = reference.get('model_config')
        if recorded is not None:
            mismatch = [key for key in model.config if key in recorded and model.config[key] != recorded[key]]
            if mismatch:
                raise ValueError(f'Checkpoint model configuration differs: {mismatch}')
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError('Checkpoint tensors are incompatible with world_model.yaml model settings: '+str(error)) from error
        model.reference = reference
        return model

    @classmethod
    def from_config(cls, directory=None, *, device=None, precision=None):
        documents, navigation = load_agent_config() if directory is None else load_agent_config(directory)
        doc = documents['world_model']
        runtime = doc['runtime']
        selected_device = device or runtime['device']
        if selected_device == 'auto':
            selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        selected_precision = precision or runtime['precision']
        runtime.update(device=selected_device, precision=selected_precision)
        weights = doc['weights']
        if weights['checkpoint'] is None:
            if weights['encoder_reference'] is not None:
                raise ValueError('For initialization without a world checkpoint, use AgentTrain.from_initialization explicitly')
            model = cls(doc['model'], device=selected_device, precision=selected_precision, navigation=documents['navigation'])
        else:
            model = cls.from_checkpoint(weights['checkpoint'], encoder_reference=weights['encoder_reference'],
                checkpoint_sha256=weights['checkpoint_sha256'], encoder_sha256=weights['encoder_sha256'],
                config=doc['model'], device=selected_device, precision=selected_precision, navigation=documents['navigation'])
        model.navigation_config = navigation
        model.resolved_config = documents
        return model

    def save_config(self, directory):
        """Write the three resolved YAML documents; no training settings are added."""
        if self.resolved_config is None:
            raise RuntimeError('save_config requires an Agent constructed with from_config')
        save_resolved_config(directory, self.resolved_config)

    def initialize_h(self, batch_size=1):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, self.inner_token_count, self.inner_state_dim, device=self.device, dtype=torch.float32)

    def _predict_visual(self, state):
        queries = self.visual_query_norm(self.visual_queries.expand(len(state), -1, -1))
        source = self.visual_source_norm(state)
        value = self.visual_cross_attention(queries, source, source, need_weights=False)[0]
        return self.visual_output_projection(self.visual_output_norm(value))

    def _predict_coordinate(self, state):
        return self.joint_readout(state)["marginal"]

    def _predict_collision(self, state):
        return self.joint_readout(state)["collision_logit"]

    @torch.no_grad()
    def encode(self, observation):
        with self.autocast():
            tokens = self.encoder(observation)
        # Ordinary detached tensors can also be consumed by AgentTrain's autograd graph.
        return tokens.clone()

    def observe(self, state, visual, coordinates):
        """Differentiable low-level O; coordinates are origin-relative patch units."""
        return self.observation.correct(state, visual, coordinates=coordinates)

    @torch.no_grad()
    def predict(self, state):
        with self.autocast():
            joint = self.joint_readout(state)
            visual = self._predict_visual(state)
            reachability = self.action_readout(state)
        return {"joint": joint, "coordinate_logits": joint["marginal"],
                "collision_logit": joint["collision_logit"],
                "predicted_visual_tokens": visual, "reachability": reachability}

    @torch.no_grad()
    def reset(self, observation, *, goal=None, branch_seed=None, execution_seed=None, episode_limit=None):
        """Initialize H+ with an observation at the episode origin (relative 0,0)."""
        visual = self.encode(observation)
        self.relative_position = torch.zeros(len(visual), 2, device=self.device, dtype=torch.long)
        with self.autocast():
            self.h = self.observe(self.initialize_h(len(visual)), visual, self.relative_position)
        self._navigation = None
        if goal is not None:
            self.start_navigation(goal, branch_seed=branch_seed, execution_seed=execution_seed,
                                  episode_limit=episode_limit)
        return self.h.clone()

    @torch.no_grad()
    def step(self, action_ids, observation, actual_displacement):
        """D then O; the external accumulator uses actual movement, including collision=0."""
        if self.h is None:
            raise RuntimeError("Call reset before step")
        actions = self._actions(action_ids, len(self.h))
        delta = torch.as_tensor(actual_displacement, device=self.device)
        if delta.shape == (2,) and len(self.h) == 1:
            delta = delta[None]
        if delta.shape != (len(self.h), 2) or delta.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actual_displacement must be integer patch units [B,2]")
        commanded = torch.tensor(MOVES, device=self.device)[actions]
        if not bool(((delta == 0).all(-1) | (delta == commanded).all(-1)).all()):
            raise ValueError("Atomic patch movement must be zero or the commanded move")
        if self._navigation is not None:
            navigation = self._navigation
            action = int(actions[0])
            if navigation.ledger.ended or navigation.halt is not None:
                raise RuntimeError("Navigation has ended; reset before another step")
            if navigation.prepared_step != navigation.ledger.steps or action != self._navigation_action():
                raise ValueError("Execute the action returned by decide() before supplying feedback")
            prior, result = self._navigation_feedback(action, observation, delta[0].cpu().numpy(), bool((delta[0] == 0).all()))
            self.relative_position = torch.tensor([navigation.ledger.position], device=self.device, dtype=torch.long)
            return {"h_prior": prior, "h": self.h.clone(),
                    "relative_position": self.relative_position.clone(), "navigation": result}
        coordinates = self.relative_position + delta
        visual = self.encode(observation)
        with self.autocast():
            prior = self.dynamic(self.h, actions)
            posterior = self.observe(prior, visual, coordinates)
        self.relative_position, self.h = coordinates, posterior
        return {"h_prior": prior, "h": posterior.clone(), "relative_position": coordinates.clone()}

    def _actions(self, actions, batch_size):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.ndim == 0 and batch_size == 1:
            actions = actions[None]
        if actions.shape != (batch_size,) or actions.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Actions must be integer [B]")
        if bool(((actions < 0) | (actions >= 8)).any()):
            raise ValueError("Actions must lie in [0,7]")
        return actions.long()

    @torch.no_grad()
    def dream_rollout(self, action_sequences, *, state=None):
        """Pure D rollout: no image, coordinate, map, or true-motion feedback is accepted."""
        root = self.h if state is None else state
        if root is None:
            raise RuntimeError("Call reset or pass a root state")
        actions = torch.as_tensor(action_sequences, device=self.device)
        if actions.ndim == 1:
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("Expected nonempty actions [N,T]")
        if len(root) == 1:
            root = root.expand(len(actions), -1, -1)
        if len(root) != len(actions):
            raise ValueError("Root batch must equal action batch, or be a single shared root")
        h = root.clone()
        states, outputs = [], []
        for action in actions.unbind(1):
            with self.autocast():
                h = self.dynamic(h, self._actions(action, len(h)))
            states.append(h)
            outputs.append(self.predict(h))
        result = {"h": torch.stack(states, 1)}
        for key in ("coordinate_logits", "collision_logit", "predicted_visual_tokens", "reachability"):
            result[key] = torch.stack([out[key] for out in outputs], 1)
        result["joint"] = {key: torch.stack([out["joint"][key] for out in outputs], 1) for key in outputs[0]["joint"]}
        return result

    def forward(self, state, action_ids):
        """Differentiable low-level Dynamic; high-level inference methods use no_grad."""
        return self.dynamic(state, action_ids)

    @staticmethod
    def _new_navigation(policy, goal, limit):
        """Data container only: all navigation behavior is implemented on Agent."""
        return SimpleNamespace(c=policy, ledger=Ledger(goal, limit, policy),
                               table=PointTable(goal, policy), queue=None, cursor=0,
                               decisions=0, halt=None, previous_hits=None, prepared_step=-1,
                               recent_positions=[(0, 0)])

    def _navigation_sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _navigation_dynamic(self, state, actions):
        with self.autocast():
            return self.dynamic(state, actions).float()

    @torch.no_grad()
    def _navigation_scores(self, state):
        with self.autocast():
            scores = self.action_readout(state)
        return scores.float().cpu().numpy()

    @torch.no_grad()
    def _navigation_decode(self, state, positions, actions):
        position = torch.as_tensor(positions, device=self.device, dtype=torch.long)
        motion = torch.as_tensor(_MOVES[np.asarray(actions)], device=self.device, dtype=torch.long)
        with self.autocast():
            joint = self.joint_readout(state)
        return self.joint_readout.decode(joint, position, motion)

    @torch.no_grad()
    def _navigation_probe(self):
        actions = np.arange(8)
        prior = self._navigation_dynamic(self.h.expand(8, -1, -1).clone(), torch.as_tensor(actions, device=self.device))
        decoded = self._navigation_decode(prior, [self._navigation.ledger.position]*8, actions)
        return {"predicted_hits": decoded["collision"].cpu().bool().tolist(),
                "positions": decoded["position"].cpu().tolist()}

    @torch.no_grad()
    def _navigation_feedback(self, action, frame, displacement, hit):
        navigation = self._navigation
        assert action == self._navigation_action()
        before, step = navigation.ledger.position, navigation.ledger.steps
        expected_pos = navigation.queue["positions"][navigation.cursor]
        expected_hit = navigation.queue["hits"][navigation.cursor]
        position = np.asarray(before) + displacement
        prior = self._navigation_dynamic(self.h, torch.tensor([action], device=self.device))
        predicted = self._navigation_decode(prior, [before], [action])
        with self.autocast():
            visual = self.encoder(frame).clone()
            coordinates = torch.as_tensor(position, device=self.device, dtype=torch.long).reshape(-1, 2)
            posterior = self.observe(prior, visual, coordinates).float()
        if not bool(torch.isfinite(posterior).all()):
            raise FloatingPointError("Nonfinite observed state")
        self.h = posterior.detach()
        result = navigation.ledger.append(action, position, hit)
        result.update(predicted_position=predicted["position"][0].cpu().tolist(),
                      predicted_hit=bool(predicted["collision"][0]))
        navigation.recent_positions.append(navigation.ledger.position)
        navigation.recent_positions = navigation.recent_positions[-navigation.c["registration"]["window"]:]
        gain, repel = potential_components(before, navigation.ledger.position, navigation.ledger.goal,
                                           navigation.table.records, step, navigation.c)
        components = [float(result["success"])*navigation.c["reward"]["success"], float(gain), float(repel)]
        result.update(reward=sum(components), components=components)
        navigation.cursor += 1
        mismatch = tuple(expected_pos) != navigation.ledger.position or bool(expected_hit) != bool(hit)
        reason = segment_end(navigation.ledger, hit, mismatch and navigation.c['execution']['replan_on_prediction_mismatch'], navigation.cursor, navigation.queue["execute_length"])
        result.update(plan_id=navigation.queue["plan_id"], segment_offset=navigation.cursor,
                      segment_end=reason, expected_position=expected_pos, expected_hit=bool(expected_hit),
                      prediction_mismatch=mismatch, field_step=step)
        if reason is not None:
            navigation.queue, navigation.cursor = None, 0
        return prior, result

    def _navigation_runtime_state(self):
        navigation = self._navigation
        return dict(h=self.h.cpu().clone(), ledger=navigation.ledger.state(), table=navigation.table.state(),
                    queue=deepcopy(navigation.queue), cursor=navigation.cursor, decisions=navigation.decisions,
                    halt=navigation.halt, previous_hits=deepcopy(navigation.previous_hits),
                    prepared_step=navigation.prepared_step, recent_positions=deepcopy(navigation.recent_positions))

    def start_navigation(self, goal, *, branch_seed=None, execution_seed=None, episode_limit=None):
        """Attach final navigation at a just-reset state; goal is origin-relative [x,y]."""
        if self.reference is None:
            raise RuntimeError("Navigation requires a loaded world-model checkpoint")
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Navigation requires frozen eval-mode inference")
        if self.h is None or len(self.h) != 1 or self.relative_position is None:
            raise RuntimeError("Reset a single observation before starting navigation")
        if self._navigation is not None or bool(self.relative_position.ne(0).any()):
            raise RuntimeError("Reset before starting a new navigation task")
        target = np.asarray(goal)
        if target.shape != (2,) or not np.issubdtype(target.dtype, np.integer) or (abs(target) > self.coordinate_limit).any():
            raise ValueError(f"goal must be two integer origin-relative coordinates in [-{self.coordinate_limit},{self.coordinate_limit}]")
        policy = deepcopy(self.navigation_config)
        episode_limit = policy['episode_limit'] if episode_limit is None else episode_limit
        branch_seed = policy['branch_seed'] if branch_seed is None else branch_seed
        execution_seed = policy['execution_seed'] if execution_seed is None else execution_seed
        if type(episode_limit) is not int or episode_limit < 1:
            raise ValueError("episode_limit must be a positive integer")
        if type(branch_seed) is not int or type(execution_seed) is not int or min(branch_seed, execution_seed) < 0:
            raise ValueError('Navigation seeds must be nonnegative integers')
        policy.update(episode_limit=episode_limit, branch_seed=branch_seed, execution_seed=execution_seed)
        navigation = self._new_navigation(policy, target, episode_limit)
        navigation.branch_rng = np.random.default_rng(branch_seed)
        navigation.execution_rng = np.random.default_rng(execution_seed)
        navigation.ledger.ended = navigation.ledger.goal == (0, 0)
        self._navigation = navigation

    @torch.no_grad()
    def decide(self):
        """Guard/replan as needed and return one action; never step the environment.

        Calling again before step is idempotent and consumes no additional random
        numbers. action=None means success, timeout or no safe candidate.
        """
        navigation = self._navigation
        if navigation is None:
            raise RuntimeError("Call reset(image, goal=...) or start_navigation(goal) first")
        ledger = navigation.ledger
        if ledger.ended or navigation.halt is not None:
            status = ("success" if ledger.position == ledger.goal else
                      "timeout" if ledger.remaining == 0 else navigation.halt)
            return {"action": None, "status": status, "step": ledger.steps,
                    "position": list(ledger.position), "guard": None, "plans": []}
        guard, plans = self._navigation_prepare(navigation.branch_rng, navigation.execution_rng)
        action = None if navigation.halt is not None else self._navigation_action()
        return {"action": action, "status": navigation.halt or "active", "step": ledger.steps,
                "position": list(ledger.position), "guard": guard, "plans": plans}

    def navigation_state(self):
        """Compact runtime snapshot: H, queues, point/history memory and both RNGs."""
        from copy import deepcopy
        if self._navigation is None:
            raise RuntimeError("No navigation task is active")
        navigation = self._navigation
        return {"format_version": "wall_world.navigation.v2", "policy_sha256": digest(navigation.c),
                "base_policy_sha256": digest(self.navigation_config), "navigation_config": deepcopy(navigation.c),
                "model_config_sha256": digest(self.config),
                "checkpoint_sha256": self.reference["checkpoint_sha256"], "precision": self.precision,
                "state": self._navigation_runtime_state(),
                "branch_rng": deepcopy(navigation.branch_rng.bit_generator.state),
                "execution_rng": deepcopy(navigation.execution_rng.bit_generator.state)}

    def restore_navigation(self, snapshot):
        """Restore onto the same frozen E6/precision, without observations or maps."""
        from copy import deepcopy
        if self.reference is None or self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Restore requires a loaded frozen inference model")
        if (snapshot.get("format_version") != "wall_world.navigation.v2"
                or snapshot.get("base_policy_sha256") != digest(self.navigation_config)
                or snapshot.get("policy_sha256") != digest(snapshot.get('navigation_config'))
                or snapshot.get("model_config_sha256") != digest(self.config)
                or snapshot.get("checkpoint_sha256") != self.reference.get("checkpoint_sha256")
                or snapshot.get("precision") != self.precision):
            raise ValueError("Navigation snapshot model, policy or precision differs")
        state = snapshot["state"]
        if state["h"].shape != (1, self.inner_token_count, self.inner_state_dim) or not bool(torch.isfinite(state["h"]).all()):
            raise ValueError("Snapshot H differs from configured state shape or is nonfinite")
        policy = deepcopy(self.navigation_config)
        policy.update({key:snapshot['navigation_config'][key] for key in ('episode_limit','branch_seed','execution_seed')})
        if policy != snapshot['navigation_config'] or state['ledger']['limit'] != policy['episode_limit']:
            raise ValueError('Snapshot navigation configuration differs')
        navigation = self._new_navigation(policy, state["ledger"]["goal"], state["ledger"]["limit"])
        navigation.ledger = Ledger.restore(state["ledger"], navigation.c)
        navigation.table.restore(state["table"])
        for key in ("queue", "cursor", "decisions", "halt", "previous_hits", "prepared_step"):
            setattr(navigation, key, deepcopy(state[key]))
        navigation.recent_positions = [tuple(p) for p in state["recent_positions"]]
        if not (1 <= len(navigation.recent_positions) <= navigation.c["registration"]["window"]
                and navigation.recent_positions[-1] == navigation.ledger.position):
            raise ValueError("Invalid snapshot recent position history")
        navigation.branch_rng = np.random.default_rng()
        navigation.branch_rng.bit_generator.state = deepcopy(snapshot["branch_rng"])
        navigation.execution_rng = np.random.default_rng()
        navigation.execution_rng.bit_generator.state = deepcopy(snapshot["execution_rng"])
        self._navigation = navigation
        self.h = state["h"].to(self.device).float()
        self.relative_position = torch.tensor([navigation.ledger.position], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _navigation_plan(self, rng, erng, guard):
        navigation = self._navigation
        navigation.decisions += 1
        attempts, prior = [], None
        for _ in range(navigation.c['max_batches']):
            bank = self._navigation_candidates(rng)
            prior = bank['root_proposal'].tolist()
            before = deepcopy(navigation.table.records)
            scores, comp, safe = score_bank(bank, navigation.table, navigation.ledger.goal,
                                           navigation.ledger.steps, navigation.c, guard['predicted_hits'])
            best_goal = float(comp[safe, :2].sum(1).max()) if safe.any() else None
            event, evidence = registration(navigation.table, navigation.ledger.steps, scores, safe, navigation.ledger.position,
                navigation.recent_positions, navigation.c, has_safe_goal=bool(np.any(safe & np.all(
                    np.asarray(bank['positions'])[np.arange(len(safe)), bank['lengths']-1] ==
                    np.asarray(navigation.ledger.goal), axis=1))))
            components_before = comp.copy() if event is not None else None
            if event is not None:
                scores, comp, safe = score_bank(bank, navigation.table, navigation.ledger.goal,
                                               navigation.ledger.steps, navigation.c, guard['predicted_hits'])
            attempt = dict(depth=bank['depth'], lengths=bank['lengths'], first_actions=bank['actions'][:, 0],
                collision_bits=bank['collision_bits'],
                endpoints=np.take_along_axis(bank['positions'], (bank['lengths']-1)[:, None, None], axis=1)[:, 0],
                table_before=before, table_after=deepcopy(navigation.table.records),
                weights=weights(navigation.table.records, navigation.ledger.steps, navigation.c), components_before=components_before,
                components=comp, scores=scores, safe=safe, retained=int(safe.sum()), best_goal=best_goal,
                max_score_after=float(scores[safe].max()) if safe.any() else None,
                point_event=event, evidence=evidence, proposal_q=bank['proposal_q'],
                actions_sha256=bank['actions_sha256'])
            attempts.append(json_scores(attempt))
            if safe.any():
                probs = probabilities(scores, navigation.c['temperature'])
                selected = choose(probs, erng)
                length = int(bank['lengths'][selected])
                navigation.queue = dict(plan_id=navigation.decisions, actions=bank['actions'][selected, :length].tolist(),
                    positions=bank['positions'][selected, :length].tolist(), hits=bank['hits'][selected, :length].tolist(),
                    execute_length=min(length, navigation.c['execute_steps']))
                assert not any(navigation.queue['hits']) and not guard['predicted_hits'][navigation.queue['actions'][0]]
                navigation.cursor = 0
                return json_scores(dict(plan_id=navigation.decisions, step=navigation.ledger.steps, position=list(navigation.ledger.position),
                    status='selected', prior=prior, attempts=attempts, selected=selected, probabilities=probs,
                    selected_score=scores[selected], selected_is_argmax=selected == int(np.argmax(scores)),
                    chosen=deepcopy(navigation.queue)))
        navigation.halt = 'no_safe_candidate'
        return dict(plan_id=navigation.decisions, step=navigation.ledger.steps, position=list(navigation.ledger.position),
                    status=navigation.halt, prior=prior, attempts=attempts, selected=None, probabilities=None, chosen=None)

    @torch.no_grad()
    def _navigation_prepare(self,rng,erng):
        navigation = self._navigation
        if navigation.prepared_step==navigation.ledger.steps:return None,[]
        self._navigation_sync();started=time.perf_counter()
        check=self._navigation_probe() if navigation.c['guard']['enabled'] else dict(predicted_hits=[False]*self.action_count,positions=None)
        self._navigation_sync()
        check['seconds']=time.perf_counter()-started
        newly=[] if navigation.previous_hits is None else np.flatnonzero(np.asarray(navigation.previous_hits)&~np.asarray(check['predicted_hits'])).tolist()
        old=None if navigation.queue is None else dict(plan_id=navigation.queue['plan_id'],cursor=navigation.cursor,action=navigation.queue['actions'][navigation.cursor])
        reason=None
        if old is not None:
            if check['predicted_hits'][old['action']]:reason='predicted_collision'
            elif newly and navigation.c['guard']['replan_on_new_passage']:reason='new_passage'
        check.update(step=navigation.ledger.steps,position=list(navigation.ledger.position),newly_open=newly,queue_before=old,interrupt=reason)
        if reason is not None:navigation.queue=None;navigation.cursor=0
        navigation.previous_hits=deepcopy(check['predicted_hits']);plans=[]
        if navigation.queue is None and navigation.halt is None:
            self._navigation_sync();started=time.perf_counter();p=self._navigation_plan(rng,erng,check);self._navigation_sync()
            p['seconds']=time.perf_counter()-started;plans.append(p)
        navigation.prepared_step=navigation.ledger.steps
        check['executed_action']=None if navigation.halt is not None else navigation.queue['actions'][navigation.cursor]
        if check['executed_action'] is not None:assert not check['predicted_hits'][check['executed_action']]
        return check,plans

    def _navigation_action(self):
        navigation = self._navigation
        assert navigation.prepared_step==navigation.ledger.steps and navigation.queue is not None and navigation.halt is None
        return navigation.queue['actions'][navigation.cursor]

    @torch.no_grad()
    def _navigation_candidates(self, rng):
        root, ledger, config = self.h, self._navigation.ledger, self._navigation.c
        count = config['candidates']
        depth = min(config['horizon'], ledger.remaining)
        states = [ledger.clone() for _ in range(count)]
        h = root.expand(count, -1, -1).clone()
        actions, positions, hits, valid, q_records = [], [], [], [], []
        first = np.zeros(count, dtype=np.int64)
        for k in range(depth):
            q = self._navigation_scores(h)
            base = proposal_probabilities(first, k, config)
            p = proposal(base, q, config['reachability_epsilon'])
            if k == 0:
                selected = draw(torch.from_numpy(p[:config['prior_candidates']]), rng)
                action = np.r_[selected, np.arange(8)]
                first = action.copy()
            else:
                action = draw(torch.from_numpy(p), rng)
            alive = np.array([not state.ended for state in states])
            prior = self._navigation_dynamic(h, torch.as_tensor(action, device=self.device))
            decoded = self._navigation_decode(prior, [state.position for state in states], action)
            pos = decoded['position'].cpu().numpy()
            hit = decoded['collision'].cpu().numpy().astype(bool)
            for i in np.flatnonzero(alive):
                states[i].append(action[i], pos[i], hit[i])
            h = torch.where(torch.as_tensor(alive, device=self.device)[:, None, None], prior, h)
            q_records.append(q)
            positions.append(np.array([state.position for state in states]))
            actions.append(action.copy())
            hits.append(hit)
            valid.append(alive)
        valid, hits = np.stack(valid, 1), np.stack(hits, 1)
        actions = np.stack(actions, 1)
        return dict(root=list(ledger.position), actions=actions, positions=np.stack(positions, 1),
                    hits=hits, valid=valid, lengths=valid.sum(1),
                    collision_bits=((hits & valid).astype(np.int64)*(1 << np.arange(depth))).sum(1),
                    depth=depth, proposal_q=pack_scores(np.stack(q_records)),
                    actions_sha256=hashlib.sha256(actions.astype(np.uint8).tobytes()).hexdigest(),
                    root_proposal=proposal(np.full((1, 8), 1/8), q_records[0][:1], config['reachability_epsilon'])[0])
