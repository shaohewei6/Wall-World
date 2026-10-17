"""EXP174 K6 objective and explicit, resumable offline training CLI."""
if __name__ == '__main__' and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parent))

import math
import torch
from torch.nn import functional as F

if __package__:
    from .agent import Agent
    from .util.losses import gaussian_targets, action_items, js
    from .util.protocol import mode_batch, coordinate_variance
    from .util.checkpoint import load_initialization, ENCODER_SHA256
    from .model.readout import CoordinateAttentionHead, JointReadout, ActionReadout
else:
    from agent import Agent
    from util.losses import gaussian_targets, action_items, js
    from util.protocol import mode_batch, coordinate_variance
    from util.checkpoint import load_initialization, ENCODER_SHA256
    from model.readout import CoordinateAttentionHead, JointReadout, ActionReadout


class AgentTrain(Agent):
    """Train world/readout parameters; DINOv2 and its LoRA remain frozen."""
    def __init__(self, config=None, *, device="cpu", precision="float32", navigation=None):
        super().__init__(config, device=device, precision=precision, navigation=navigation)
        self.requires_grad_(True)
        self.encoder.requires_grad_(False)
        self.train()

    @classmethod
    def from_initialization(cls, encoder_reference, *, config=None, device="cpu", precision="float32", navigation=None):
        """Reproduce original EXP174 step-zero tensors, including its seeded new heads.

        This only assembles/loads a trainable model. It does not create an optimizer
        or run training. A bare constructor has random weights; prefer this factory
        or from_checkpoint so the frozen visual encoder is properly pretrained.
        """
        initial = load_initialization(encoder_reference)
        model = cls(config, device=device, precision=precision, navigation=navigation)
        c = model.config
        coordinate = CoordinateAttentionHead(c['inner_state_dim'], c['coordinate_attention_heads'], c['coordinate_hidden_dim'], c['coordinate_limit']).to(device)
        coordinate.load_state_dict({name.removeprefix("current_coordinate."): value
                                    for name, value in initial.items() if name.startswith("current_coordinate.")}, strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(202610131)
            model.joint_readout = JointReadout(coordinate)
            model.action_readout = ActionReadout(c['inner_state_dim'], c['action_attention_heads'], c['action_hidden_dim'], c['action_count'])
        model.to(device)
        combined = model.state_dict()
        combined.update({name: value for name, value in initial.items()
                         if not name.startswith(("current_coordinate.", "collision_head."))})
        model.load_state_dict(combined, strict=True)
        model.reference = {"initialization": "EXP174 canonical step0", "global_step": 0,
                           "encoder_sha256": ENCODER_SHA256}
        return model

    @staticmethod
    def _reduce_loss(loss, reduction):
        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError("reduction must be none, mean or sum")

    @staticmethod
    def visual_target(visual):
        return F.layer_norm(visual.detach().float(), (visual.shape[-1],))

    def compute_visual_loss(self, prediction, target, *, reduction="mean"):
        loss = (prediction.float() - self.visual_target(target)).abs().mean((-2, -1))
        return self._reduce_loss(loss, reduction)

    def compute_coordinate_loss(self, logits, coordinates, *, reduction="mean"):
        if logits.shape[:-1] != coordinates.shape or logits.shape[-2:] != (2, self.config['coordinate_classes']):
            raise ValueError("Coordinate logits differ from configured vocabulary or coordinates")
        target = gaussian_targets(coordinates, self.coordinate_sigma, self.coordinate_limit)
        # Slice PAD before multiplication: zero times -inf would produce NaN.
        loss = -(target * F.log_softmax(logits[..., 1:].float(), -1)).sum(-1).mean(-1)
        return self._reduce_loss(loss, reduction)

    def compute_collision_loss(self, logits, flags, *, reduction="mean"):
        if flags.dtype != torch.bool or logits.shape != flags.shape:
            raise ValueError("Collision must be a boolean label with the same shape")
        loss = F.binary_cross_entropy_with_logits(logits.float(), flags.float(), reduction="none")
        return self._reduce_loss(loss, reduction)

    def conditional_loss(self, joint, coordinate, collision, reduction="none"):
        branch = joint["conditional"][torch.arange(len(coordinate), device=coordinate.device), collision.long()]
        return self.compute_coordinate_loss(branch, coordinate, reduction=reduction)

    def read_losses(self, state, visual, coordinates, collision):
        joint = self.joint_readout(state)
        prediction = self._predict_visual(state)
        return {"coordinate": self.conditional_loss(joint, coordinates, collision),
                "visual": self.compute_visual_loss(prediction, visual, reduction="none"),
                "collision": self.compute_collision_loss(joint["collision_logit"], collision, reduction="none")}

    def forward(self, state, visual, *, coordinates):
        """Observation and differentiable readouts (same low-level interface as EXP174)."""
        posterior = self.observe(state, visual, coordinates)
        return {"h_o": posterior, "coordinate_logits": self._predict_coordinate(posterior),
                "predicted_visual_tokens": self._predict_visual(posterior)}

    def sequence_loss(self, values, *, state=None, chunk_starts=None, observe_mask=None,
                      depth_before=None, episode_ids=None, seed=20260918, epoch=0,
                      global_step=0, return_graph=False):
        """Original K6 objective, preserving state across caller-managed chunks.

        values: actions [B,T], visual [B,T+1,256,384], relative_positions [B,T+1,2],
        collision_flags [B,T]. Optional policy_{action,collision,back,valid} are [B,T+1]
        and label the NEXT executed action at each state. The final label may be invalid.
        observe_mask is True only when real O follows that transition; D steps never
        compute a shadow O. Otherwise an episode-global uniform-length-1..6 schedule
        is generated from episode_ids, chunk_starts, seed and epoch.
        """
        actions, visual, coords, collision = [values[k] for k in
                                            ("actions", "visual", "relative_positions", "collision_flags")]
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("actions must be nonempty [B,T]")
        batch, steps = actions.shape
        if visual.shape != (batch, steps + 1, self.config['visual_token_count'], self.config['visual_state_dim']) or coords.shape != (batch, steps + 1, 2):
            raise ValueError("Visual/coordinate sequences must have T+1 frames")
        if collision.shape != (batch, steps) or collision.dtype != torch.bool:
            raise ValueError("collision_flags must be boolean [B,T]")
        if any(x.device != self.device for x in (actions, visual, coords, collision)):
            raise ValueError("All sequence tensors must be on the model device")
        starts = torch.zeros(batch, dtype=torch.long, device=self.device) if chunk_starts is None else torch.as_tensor(chunk_starts, device=self.device)
        if starts.shape != (batch,) or starts.dtype not in (torch.int32, torch.int64) or bool((starts < 0).any()):
            raise ValueError("chunk_starts must be nonnegative integer [B]")
        if observe_mask is None:
            if episode_ids is None:
                raise ValueError("Supply observe_mask/depth_before or episode_ids for the K6 schedule")
            observe_mask, depth_before = mode_batch(episode_ids, starts.cpu().tolist(), steps,
                                                  seed=seed, epoch=epoch, maximum=6)
        elif depth_before is None:
            if bool((starts != 0).any()):
                raise ValueError("Continuing chunks require depth_before")
            depth_before = torch.zeros(batch, dtype=torch.long)
        mask = torch.as_tensor(observe_mask, device=self.device)
        depth = torch.as_tensor(depth_before, device=self.device)
        if mask.shape != (batch, steps) or mask.dtype != torch.bool:
            raise ValueError("observe_mask must be boolean [B,T]")
        if depth.shape != (batch,) or depth.dtype not in (torch.int32, torch.int64) or bool((depth < 0).any()):
            raise ValueError("depth_before must be nonnegative integer [B]")
        initial = starts == 0
        if state is None and not bool(initial.all()):
            raise ValueError("Continuing chunks require their carried state")
        state = self.initialize_h(batch) if state is None else state
        if state.shape != (batch, self.inner_token_count, self.inner_state_dim) or state.device != self.device:
            raise ValueError("state must match the configured token count, width and model device")
        if bool(state[initial].ne(0).any()):
            raise ValueError("Episode starts require zero H")
        self.coordinate_sigma = math.sqrt(coordinate_variance(global_step) if self.training else .5)
        policy_keys = ("policy_action", "policy_collision", "policy_back", "policy_valid")
        policy_enabled = any(k in values for k in policy_keys)
        if policy_enabled:
            if not all(k in values and values[k].shape == (batch, steps + 1) and values[k].device == self.device for k in policy_keys):
                raise ValueError("All four policy label tensors must be [B,T+1] on the model device")
            if values["policy_collision"].dtype != torch.bool or values["policy_valid"].dtype != torch.bool:
                raise ValueError("Policy collision/valid flags must be boolean")
        physical = {key: [] for key in ("coordinate", "visual", "collision")}
        policies, alignments, valids = [], [], []

        def correct(selected, frame, prior):
            if not bool(selected.any()):
                return prior
            posterior = self.observe(prior[selected], visual[selected, frame], coords[selected, frame])
            result = prior.clone()
            result[selected] = posterior
            return result

        def policy(frame, prior, posterior, observed, is_initial=False):
            if not policy_enabled:
                return
            valid = values["policy_valid"][:, frame]
            if is_initial:
                valid = valid & observed
            action = values["policy_action"][:, frame]
            hit = values["policy_collision"][:, frame]
            back = values["policy_back"][:, frame]
            logits = self.action_readout(prior, return_logits=True)
            items, _ = action_items(logits, action, hit, back)
            align = torch.zeros(batch, device=self.device)
            selected = observed & valid
            if bool(selected.any()) and not is_initial:
                post_logits = self.action_readout(posterior[selected], return_logits=True)
                post_items, _ = action_items(post_logits, action[selected], hit[selected], back[selected])
                items = items.clone()
                items[selected] = (items[selected] + post_items) / 2
                align[selected] = js(logits[selected].sigmoid(), post_logits.sigmoid())
            policies.append(items[valid].sum())
            alignments.append(align[valid].sum())
            valids.append(valid.sum())

        with self.autocast():
            if bool(initial.any()):
                state = correct(initial, 0, state)
                policy(0, state, state, initial, True)
            for time in range(steps):
                depth = depth + 1
                if int(depth.max()) > 6:
                    raise ValueError("K6 schedule exceeded six consecutive Dynamic predictions")
                prior = self.dynamic(state, self._actions(actions[:, time], batch))
                losses = self.read_losses(prior, visual[:, time + 1], coords[:, time + 1], collision[:, time])
                for key in physical:
                    physical[key].append(losses[key])
                posterior = correct(mask[:, time], time + 1, prior)
                policy(time + 1, prior, posterior, mask[:, time])
                state = posterior
                depth = torch.where(mask[:, time], 0, depth)
            components = {key: torch.cat(value).mean() for key, value in physical.items()}
            denominator = sum(valids).clamp_min(1) if valids else torch.ones((), device=self.device)
            components["action"] = sum(policies) / denominator if policies else state.sum() * 0
            components["align"] = sum(alignments) / denominator if alignments else state.sum() * 0
            total = sum(components[key] for key in physical) + .2 * components["action"] + .05 * components["align"]
        if not bool(torch.isfinite(total)) or not bool(torch.isfinite(state).all()):
            raise FloatingPointError("Nonfinite K6 objective or state")
        return {"loss": total, "components": components,
                "state": state if return_graph else state.detach(), "depth": depth.detach()}


def train_offline(args, *, training_document=None):
    """Stateful episode batches with TBPTT, frozen online encoding and AdamW.

    max_steps is an absolute optimizer-step target. Resume restores H and the
    episode/chunk cursor as well as optimizer, scheduler and RNG. This is a new
    training stream format, not a replacement for historical EXP174 resume.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import random
    import signal
    import time
    import uuid
    import numpy as np
    import yaml
    from tqdm import tqdm
    if __package__:
        from .util.config import load_agent_config, save_resolved_config
        from .util.offline_data import OfflineEpisodes
        from .util.checkpoint import atomic_save, save_training_state, load_training_state, sha256
    else:
        from util.config import load_agent_config, save_resolved_config
        from util.offline_data import OfflineEpisodes
        from util.checkpoint import atomic_save, save_training_state, load_training_state, sha256

    root = Path(__file__).resolve().parent
    output = Path(args.output).expanduser().resolve()
    if output.is_relative_to(root/'checkpoints'):
        raise ValueError('Training outputs cannot overwrite packaged checkpoints')
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise ValueError('Use a new/empty output directory, or explicitly --resume')
    if args.resume is not None and not output.is_dir():
        raise ValueError('Resume requires the existing run output directory')
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    documents, _ = load_agent_config(args.config_dir)
    wm = documents['world_model']
    device = args.device or wm['runtime']['device']
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    precision = args.precision or wm['runtime']['precision']
    wm['runtime'].update(device=device, precision=precision)
    if training_document is not None:
        from copy import deepcopy
        training_document = deepcopy(training_document)
        training_document['runtime'].update(device=device,precision=precision)
    weights = wm['weights']
    reference = weights['encoder_reference']
    if reference is None:
        raise ValueError('Offline training requires an explicit frozen encoder_reference asset')
    reference_sha = sha256(reference)
    if weights['encoder_sha256'] is not None and reference_sha != weights['encoder_sha256']:
        raise ValueError('Frozen encoder reference SHA256 mismatch')
    world = documents['world']['geometry']
    if not Path(args.data).is_dir():
        raise FileNotFoundError(f'Offline training data not found: {args.data}. '
            'Set train.yaml data.root or --data to a recorded-trajectory directory.')
    dataset = OfflineEpisodes(args.data, image_size=wm['model']['image_size'], patch_size=world['patch_size'],
        grid_size=world['map_tiles_per_axis']*world['tile_size']//world['patch_size'],
        episode_ids=args.episode_ids, cache_size=args.batch_size)
    groups = math.ceil(len(dataset.ids)/args.batch_size)
    source_paths = [root/'agent.py',root/'agent_train.py',*sorted((root/'model').rglob('*.py')),
                    *sorted((root/'util').glob('*.py'))]
    source_hashes = {str(p.relative_to(root)):sha256(p) for p in source_paths}
    contract = dict(model=wm['model'], dataset_fingerprint=dataset.fingerprint,
        episode_ids=dataset.ids, batch_size=args.batch_size, chunk_length=args.chunk_length,
        encoder_batch_size=args.encoder_batch_size, encoder_sha256=reference_sha,
        device=str(torch.device(device)), precision=precision, seed=args.seed,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        betas=list(args.betas), eps=args.eps,
        max_grad_norm=args.max_grad_norm, warmup_steps=args.warmup_steps,
        schedule_steps=args.schedule_steps, minimum_learning_rate=args.minimum_learning_rate,
        source_sha256=source_hashes)
    saved = None
    legacy_cli_resume = False
    if args.resume is not None:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        stored_contract = saved.get('stream',{}).get('contract')
        # A narrowly identified CLI-only predecessor used the same update loop,
        # fixed default AdamW betas/eps, and exactly these two source revisions.
        legacy_sources = dict(source_hashes)
        legacy_sources.update({'agent_train.py':'c99da964c2ac30c6227154accf67455cb7e4fafa208cc5a5616d90f0e82434a8',
                               'util/config.py':'613fdfe891a7202a2195c6a10d90f527d24abdf809186e985a917fe875689437'})
        if isinstance(stored_contract,dict) and stored_contract.get('source_sha256')==legacy_sources:
            stored_contract = dict(stored_contract,source_sha256=source_hashes)
            stored_contract.setdefault('betas',[0.9,0.999]);stored_contract.setdefault('eps',1e-8)
            legacy_cli_resume = True
        if saved.get('format_version') != 'wall_world.training.v1' or stored_contract != contract:
            raise ValueError('Resume model/data/training settings/source differ from the saved run')
        if saved['global_step'] >= args.max_steps:
            raise ValueError('max_steps must exceed the number of completed optimizer steps')
        model = AgentTrain(wm['model'],device=device,precision=precision,navigation=documents['navigation'])
    elif args.initialization == 'canonical':
        model = AgentTrain.from_initialization(reference,config=wm['model'],device=device,precision=precision,
                                               navigation=documents['navigation'])
    else:
        if weights['checkpoint'] is None: raise ValueError('checkpoint initialization needs weights.checkpoint')
        model = AgentTrain.from_checkpoint(weights['checkpoint'],encoder_reference=reference,
            config=wm['model'],device=device,precision=precision,navigation=documents['navigation'],
            checkpoint_sha256=weights['checkpoint_sha256'],encoder_sha256=reference_sha)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=args.weight_decay,
                                 betas=tuple(args.betas),eps=args.eps)
    def factor(step):
        if step < args.warmup_steps: return (step+1)/max(1,args.warmup_steps)
        progress = min(1.,(step-args.warmup_steps)/max(1,args.schedule_steps-args.warmup_steps))
        minimum = args.minimum_learning_rate/args.learning_rate
        return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    def parameter_digest(module, *, encoder):
        h = hashlib.sha256()
        for name,tensor in module.state_dict().items():
            if name.startswith('encoder.') != encoder: continue
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    if saved is not None:
        restored = load_training_state(args.resume,model,optimizer,scheduler=scheduler,encoder_reference=reference)
        stream = restored['stream']; step = restored['global_step']
        state = None if stream['carried_h'] is None else stream['carried_h'].to(model.device)
        epoch, group, start = stream['epoch'],stream['group'],stream['chunk_start']
        initial_hash, encoder_hash = stream['initial_model_sha256'],stream['encoder_tensor_sha256']
        run_id = stream['run_id']
        del saved
    else:
        step = epoch = group = start = 0; state = None
        initial_hash = parameter_digest(model,encoder=False)
        encoder_hash = parameter_digest(model,encoder=True)
        run_id = uuid.uuid4().hex
    encoder_versions = {n:p._version for n,p in model.encoder.named_parameters()}
    if args.resume is None and args.initialization == 'checkpoint':
        reference_state = torch.load(reference,map_location='cpu',weights_only=False)['model']
        for name,tensor in model.encoder.state_dict().items():
            if not torch.equal(tensor.detach().cpu(),reference_state['encoder.'+name]):
                raise ValueError('Checkpoint encoder differs from the reference used for compact saving')
        del reference_state
    assert model.training and not model.encoder.training and parameters
    output.mkdir(parents=True,exist_ok=True)
    if args.resume is None:
        save_resolved_config(output/'config',documents)
        (output/'training.yaml').write_text(yaml.safe_dump(dict(vars(args),run_id=run_id),sort_keys=False))
        (output/'data_manifest.yaml').write_text(yaml.safe_dump(dict(root=str(dataset.root),
            fingerprint=dataset.fingerprint,episodes=dataset.inventory),sort_keys=False))
    attempt = uuid.uuid4().hex
    if training_document is not None:
        (output/f'train_config_{attempt}.yaml').write_text(yaml.safe_dump(training_document,sort_keys=False))
        if args.resume is None:
            (output/'config/train.yaml').write_text(yaml.safe_dump(training_document,sort_keys=False))
    def append(name,row):
        with (output/name).open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
    append('attempts.jsonl',dict(attempt=attempt,run_id=run_id,start_step=step,arguments=vars(args),time=time.time()))
    def status(phase, **fields):
        target = output/'status.json';temp=target.with_suffix('.json.tmp')
        row=dict(status=phase,global_step=step,epoch=epoch,group=group,
                 chunk_start=start,time=time.time(),attempt=attempt)
        row.update(fields)
        temp.write_text(json.dumps(row,indent=2)+'\n')
        os.replace(temp,target)
    def save():
        # Write the new complete checkpoint before rotating the last valid file.
        pending, latest, previous = (output/n for n in ('pending.pth','latest.pth','previous.pth'))
        stream = dict(epoch=epoch,group=group,chunk_start=start,carried_h=None if state is None else state.detach().cpu(),
            contract=contract,run_id=run_id,initial_model_sha256=initial_hash,encoder_tensor_sha256=encoder_hash)
        save_training_state(pending,model,optimizer,scheduler=scheduler,epoch=epoch,global_step=step,
            stream=stream,encoder_reference=dict(path=str(reference),sha256=reference_sha))
        if latest.exists(): os.replace(latest,previous)
        os.replace(pending,latest)
        append('checkpoints.jsonl',dict(step=step,epoch=epoch,chunk_start=start,time=time.time()))
    stop = [False]
    def request_stop(*_): stop[0] = True
    handlers = {sig:signal.signal(sig,request_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if model.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(model.device)
    last_save = time.monotonic(); completed_start = step
    print(f'Offline K6: {len(dataset.ids)} episodes, batch={args.batch_size}, chunk={args.chunk_length}, '
          f'{model.device}/{precision}, optimizer target={args.max_steps}, starting step={step}',flush=True)
    try:
        if args.resume is None: save()
        status('running')
        with tqdm(total=args.max_steps,initial=step,desc='Offline world training',mininterval=args.tqdm_mininterval,dynamic_ncols=True) as progress:
            while step < args.max_steps and not stop[0]:
                begin = time.monotonic()
                ids = dataset.group(args.seed,epoch,group,args.batch_size)
                batch, end = dataset.batch(ids,start,args.chunk_length)
                rgb = batch.pop('observations'); b,t = rgb.shape[:2]
                frames = rgb.flatten(0,1)
                if model.device.type == 'cuda': frames = frames.pin_memory()
                visual = [model.encode(frames[i:i+args.encoder_batch_size]) for i in range(0,len(frames),args.encoder_batch_size)]
                values = {k:v.to(model.device,non_blocking=True) for k,v in batch.items()}
                values['visual'] = torch.cat(visual).reshape(b,t,model.config['visual_token_count'],model.config['visual_state_dim'])
                optimizer.zero_grad(set_to_none=True)
                result = model.sequence_loss(values,state=state,chunk_starts=[start]*b,episode_ids=ids,
                                             seed=args.seed,epoch=epoch,global_step=step)
                result['loss'].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters,args.max_grad_norm,error_if_nonfinite=True)
                if any(p.grad is not None or p.requires_grad for p in model.encoder.parameters()) or model.encoder.training:
                    raise RuntimeError('Frozen encoder received gradients or entered training mode')
                lr = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                if not all(bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError('Nonfinite updated model parameter')
                row = dict(step=step+1,epoch=epoch,group=group,chunk_start=start,chunk_stop=end,
                    episode_ids=ids,loss=float(result['loss'].detach()),
                    components={k:float(v.detach()) for k,v in result['components'].items()},
                    grad_norm=float(grad_norm),learning_rate=lr,attempt=attempt)
                state = result['state']; start = end; step += 1
                if start == dataset.transitions:
                    state=None;start=0;group+=1
                    if group == groups:group=0;epoch+=1
                if model.device.type == 'cuda': torch.cuda.synchronize(model.device)
                row['seconds'] = time.monotonic()-begin
                append('metrics.jsonl',row)
                status('running',loss=row['loss'])
                progress.update(1);progress.set_postfix(loss=f"{row['loss']:.4f}",lr=f'{lr:.2g}')
                if step%args.save_every==0 or time.monotonic()-last_save >= args.save_seconds:
                    save();last_save=time.monotonic()
                del result,values,visual,rgb,frames,batch
        if step%args.save_every or step==completed_start or stop[0]: save()
        final_hash = parameter_digest(model,encoder=False)
        actual_encoder_hash = parameter_digest(model,encoder=True)
        if encoder_hash!=actual_encoder_hash or encoder_versions!={n:p._version for n,p in model.encoder.named_parameters()}:
            raise RuntimeError('Frozen encoder changed during training')
        summary = dict(global_step=step,updates_this_attempt=step-completed_start,encoder_unchanged=True,
            initial_model_sha256=initial_hash,final_model_sha256=final_hash,
            trainable_parameters_changed=initial_hash!=final_hash,encoder_tensor_sha256=encoder_hash,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(model.device) if model.device.type=='cuda' else 0,
            checkpoint=str(output/'latest.pth'),stop_requested=stop[0],run_id=run_id,
            resumed_legacy_cli_checkpoint=legacy_cli_resume)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        status('interrupted' if stop[0] else 'completed',**summary)
        print(json.dumps(summary),flush=True)
        return summary
    except BaseException:
        import traceback
        (output/f'error_{attempt}.log').write_text(traceback.format_exc())
        status('failed')
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    import argparse
    from pathlib import Path
    from types import SimpleNamespace
    if __package__:
        from .util.config import load_train, train_arguments
    else:
        from util.config import load_train, train_arguments
    parser = argparse.ArgumentParser(description='Offline K6 training from config/train.yaml; explicit CLI values override YAML')
    parser.add_argument('--train-config','--config',default=str(Path(__file__).resolve().parent/'config/train.yaml'))
    parser.add_argument('--config-dir',help='Override the directory containing inference YAML files')
    parser.add_argument('--data',help='Override the offline data root')
    parser.add_argument('--output',help='Override run output directory')
    parser.add_argument('--max-steps',type=int,help='Override absolute optimizer-step target')
    parser.add_argument('--resume',help='Override checkpoint path to resume')
    parser.add_argument('--initialization',choices=('canonical','checkpoint'))
    parser.add_argument('--episode-ids',nargs='+',type=int)
    parser.add_argument('--device')
    parser.add_argument('--precision',choices=('float32','bfloat16'))
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--chunk-length',type=int)
    parser.add_argument('--encoder-batch-size',type=int)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--weight-decay',type=float)
    parser.add_argument('--betas',nargs=2,type=float)
    parser.add_argument('--eps',type=float)
    parser.add_argument('--max-grad-norm',type=float)
    parser.add_argument('--warmup-steps',type=int)
    parser.add_argument('--schedule-steps',type=int)
    parser.add_argument('--minimum-learning-rate',type=float)
    parser.add_argument('--save-every',type=int)
    parser.add_argument('--save-seconds',type=float)
    parser.add_argument('--tqdm-mininterval',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--cpu-threads',type=int)
    selected = parser.parse_args()
    config_path = str(Path(selected.train_config).expanduser().resolve())
    overrides = {k:v for k,v in vars(selected).items() if k!='train_config' and v is not None}
    for name in ('config_dir','data','output','resume'):
        if name in overrides:overrides[name]=str(Path(overrides[name]).expanduser().resolve())
    try:
        document = load_train(config_path,overrides=overrides)
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))
    args = SimpleNamespace(**train_arguments(document),train_config=config_path)
    train_offline(args,training_document=document)


if __name__ == '__main__':
    main()
