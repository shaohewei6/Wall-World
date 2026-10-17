"""Independent inference/training YAML validation and file-relative paths."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import yaml

DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'config'
MODEL_KEYS = set('action_count image_size inner_state_dim inner_token_count dynamic_depth dynamic_heads dynamic_mlp_ratio dynamic_dropout observation_depth observation_heads observation_mlp_ratio observation_dropout visual_predictor_heads coordinate_hidden_dim coordinate_attention_heads observation_initial_gate state_target_rms lora_config encoder_type encoder_patch_size encoder_pretrain_image_size encoder_layer_scale encoder_interpolate_antialias encoder_interpolate_offset coordinate_limit visual_predictor_dropout action_attention_heads action_hidden_dim'.split())
NAV_KEYS = set('episode_limit candidates horizon execute_steps max_batches sigma_degrees reachability_epsilon temperature branch_seed execution_seed arrival_reward field registration guard execution'.split())
FIELD_KEYS = set('rho goal_strength repel_strength half_life goal_gate_length repulsion_radius initial_repulsion capacity registration_interval'.split())
REG_KEYS = set('epsilon window half_window distance_improvement repeat_fraction history_enabled no_gain_enabled'.split())
GEO_KEYS = set('viewport_size patch_size map_tiles_per_axis tile_size wall_thickness ball_size'.split())
GEN_KEYS = set('min_room_count max_room_count min_room_width max_room_width min_room_height max_room_height room_clearance min_doors_per_room max_doors_per_room door_width max_placement_attempts max_generation_attempts'.split())

class UniqueLoader(yaml.SafeLoader):
    pass

def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result
UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)

def read_yaml(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        value = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a YAML mapping')
    return value

def keys(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name}: expected a mapping')
    missing, unknown = expected - set(value), set(value) - expected
    if missing or unknown:
        raise ValueError(f'{name}: missing={sorted(missing)}, unknown={sorted(unknown)}')

def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}: expected integer >= {minimum}')

def number(value, name, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected a finite number')
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum) or (positive and value <= 0):
        raise ValueError(f'{name}: invalid range')

def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f'{name}: expected boolean')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def _asset(value, base, name):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name}: expected a nonempty path or null')
    path = Path(value).expanduser()
    return str((base/path).resolve() if not path.is_absolute() else path.resolve())


def load_train(path=DEFAULT_DIR/'train.yaml', *, overrides=None):
    """Training is explicit and separate from the three inference YAML files."""
    path = Path(path).expanduser().resolve()
    doc = read_yaml(path)
    keys(doc, {'config_dir','runtime','data','run','optimizer','scheduler','checkpoint','logging'}, 'train')
    sections = {
        'runtime': {'device','precision','seed','cpu_threads'},
        'data': {'root','episode_ids','batch_size','chunk_length','encoder_batch_size'},
        'run': {'output','max_steps','initialization','resume'},
        'optimizer': {'name','learning_rate','weight_decay','betas','eps','max_grad_norm'},
        'scheduler': {'name','warmup_steps','schedule_steps','minimum_learning_rate'},
        'checkpoint': {'save_every_steps','save_every_seconds'},
        'logging': {'tqdm_mininterval'},
    }
    for name, expected in sections.items(): keys(doc[name], expected, 'train.'+name)
    locations = {key:('runtime',key) for key in sections['runtime']}
    locations.update({key:('data',key) for key in sections['data']-{'root'}})
    locations.update({key:('run',key) for key in sections['run']})
    locations.update({key:('optimizer',key) for key in sections['optimizer']-{'name'}})
    locations.update({key:('scheduler',key) for key in sections['scheduler']-{'name'}})
    locations.update(data=('data','root'),save_every=('checkpoint','save_every_steps'),
                     save_seconds=('checkpoint','save_every_seconds'),tqdm_mininterval=('logging','tqdm_mininterval'))
    for name,value in (overrides or {}).items():
        if name=='config_dir':doc[name]=value
        elif name in locations:
            section,field=locations[name];doc[section][field]=value
        else:raise ValueError('Unknown training override: '+name)
    for section, field in [('data','root'),('run','output'),('run','resume')]:
        value = doc[section][field]
        if value is None and field!='resume': raise ValueError('train.'+field+' must be a nonempty path')
        doc[section][field] = _asset(value,path.parent,'train.'+section+'.'+field)
    if doc['config_dir'] is None: raise ValueError('train.config_dir must be a nonempty path')
    doc['config_dir'] = _asset(doc['config_dir'],path.parent,'train.config_dir')
    runtime = doc['runtime']
    if runtime['device'] is not None and (not isinstance(runtime['device'],str) or not re.fullmatch(r'cpu|auto|cuda(?::[0-9]+)?',runtime['device'])):
        raise ValueError('train.runtime.device must be cpu/auto/cuda[:index] or null')
    if runtime['precision'] not in (None,'float32','bfloat16'):
        raise ValueError('train.runtime.precision must be float32/bfloat16 or null')
    if runtime['device']=='cpu' and runtime['precision']=='bfloat16':
        raise ValueError('Use float32 for CPU training')
    integer(runtime['seed'],'train.runtime.seed',0)
    integer(runtime['cpu_threads'],'train.runtime.cpu_threads')
    for name in ('batch_size','chunk_length','encoder_batch_size'):
        integer(doc['data'][name],'train.data.'+name)
    if doc['data']['chunk_length']>4096: raise ValueError('train chunk_length exceeds the K6 schedule limit')
    ids = doc['data']['episode_ids']
    if ids is not None:
        if not isinstance(ids,list) or not ids: raise ValueError('train.data.episode_ids must be a nonempty list or null')
        for value in ids: integer(value,'train.data.episode_ids',0)
        if len(ids)!=len(set(ids)): raise ValueError('train.data.episode_ids contains duplicates')
    integer(doc['run']['max_steps'],'train.run.max_steps')
    if doc['run']['initialization'] not in ('canonical','checkpoint'):
        raise ValueError('train.run.initialization must be canonical or checkpoint')
    opt, sch = doc['optimizer'], doc['scheduler']
    if opt['name']!='adamw': raise ValueError('Only the implemented AdamW optimizer is supported')
    for name in ('learning_rate','eps','max_grad_norm'): number(opt[name],'train.optimizer.'+name,positive=True)
    number(opt['weight_decay'],'train.optimizer.weight_decay',minimum=0)
    if not isinstance(opt['betas'],list) or len(opt['betas'])!=2: raise ValueError('AdamW betas must contain two numbers')
    for value in opt['betas']:
        number(value,'train.optimizer.betas',minimum=0)
        if value>=1: raise ValueError('AdamW betas must be less than one')
    if sch['name']!='warmup_cosine': raise ValueError('Only the implemented warmup_cosine scheduler is supported')
    integer(sch['warmup_steps'],'train.scheduler.warmup_steps',0)
    integer(sch['schedule_steps'],'train.scheduler.schedule_steps')
    if sch['warmup_steps']>=sch['schedule_steps']: raise ValueError('warmup_steps must be less than schedule_steps')
    number(sch['minimum_learning_rate'],'train.scheduler.minimum_learning_rate',minimum=0,maximum=opt['learning_rate'])
    integer(doc['checkpoint']['save_every_steps'],'train.checkpoint.save_every_steps')
    number(doc['checkpoint']['save_every_seconds'],'train.checkpoint.save_every_seconds',positive=True)
    number(doc['logging']['tqdm_mininterval'],'train.logging.tqdm_mininterval',positive=True)
    return doc


def train_arguments(doc):
    """Flatten validated training settings for the offline loop; no second defaults."""
    r,d,run,o,s,c,l = (doc[k] for k in ('runtime','data','run','optimizer','scheduler','checkpoint','logging'))
    return dict(config_dir=doc['config_dir'],data=d['root'],output=run['output'],max_steps=run['max_steps'],
        resume=run['resume'],initialization=run['initialization'],episode_ids=d['episode_ids'],
        device=r['device'],precision=r['precision'],seed=r['seed'],cpu_threads=r['cpu_threads'],
        batch_size=d['batch_size'],chunk_length=d['chunk_length'],encoder_batch_size=d['encoder_batch_size'],
        learning_rate=o['learning_rate'],weight_decay=o['weight_decay'],betas=o['betas'],eps=o['eps'],
        max_grad_norm=o['max_grad_norm'],warmup_steps=s['warmup_steps'],schedule_steps=s['schedule_steps'],
        minimum_learning_rate=s['minimum_learning_rate'],save_every=c['save_every_steps'],
        save_seconds=c['save_every_seconds'],tqdm_mininterval=l['tqdm_mininterval'])

def load_world_model(path=DEFAULT_DIR/'world_model.yaml'):
    path = Path(path).expanduser().resolve()
    doc = read_yaml(path)
    keys(doc, {'runtime', 'weights', 'model'}, 'world_model')
    keys(doc['runtime'], {'device', 'precision'}, 'runtime')
    if not isinstance(doc['runtime']['device'], str) or not re.fullmatch(r'cpu|auto|cuda(?::[0-9]+)?', doc['runtime']['device']):
        raise ValueError('runtime.device: expected cpu, auto, cuda or cuda:N')
    if doc['runtime']['precision'] not in ('float32', 'bfloat16'):
        raise ValueError('runtime.precision: expected float32 or bfloat16')
    keys(doc['weights'], {'checkpoint', 'checkpoint_sha256', 'encoder_reference', 'encoder_sha256'}, 'weights')
    for name in ('checkpoint_sha256', 'encoder_sha256'):
        value = doc['weights'][name]
        if value is not None and (not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)):
            raise ValueError(f'weights.{name}: expected lowercase SHA256 or null')
    for name in ('checkpoint', 'encoder_reference'):
        doc['weights'][name] = _asset(doc['weights'][name], path.parent, 'weights.'+name)
        hash_name = 'checkpoint_sha256' if name=='checkpoint' else 'encoder_sha256'
        if doc['weights'][name] is None and doc['weights'][hash_name] is not None:
            raise ValueError(f'weights.{hash_name} requires weights.{name}')
    _validate_model(doc['model'])
    return doc

def _validate_model(c):
    keys(c, MODEL_KEYS, 'model')
    if c['encoder_type'] != 'dinov2_vits14' or c['encoder_patch_size'] != 14:
        raise ValueError('Only the implemented dinov2_vits14 backbone with patch_size=14 is supported')
    if c['action_count'] != 8:
        raise ValueError('The implemented environment/navigation action space has exactly 8 directions')
    for name in ('action_count','image_size','inner_state_dim','inner_token_count','dynamic_depth','dynamic_heads','observation_heads','visual_predictor_heads','coordinate_hidden_dim','coordinate_attention_heads','encoder_patch_size','encoder_pretrain_image_size','coordinate_limit','action_attention_heads','action_hidden_dim'):
        integer(c[name], 'model.'+name)
    integer(c['observation_depth'], 'model.observation_depth', 0)
    if c['inner_state_dim'] % 2:
        raise ValueError('inner_state_dim must be even for concatenated coordinate encoding')
    for name in ('dynamic_heads','observation_heads','visual_predictor_heads','coordinate_attention_heads','action_attention_heads'):
        if c['inner_state_dim'] % c[name]:
            raise ValueError(f'inner_state_dim must be divisible by {name}')
    for name in ('image_size','encoder_pretrain_image_size'):
        if c[name] % c['encoder_patch_size']:
            raise ValueError(f'{name} must be divisible by encoder_patch_size')
    for name in ('dynamic_mlp_ratio','observation_mlp_ratio','state_target_rms','encoder_layer_scale'):
        number(c[name], 'model.'+name, positive=True)
    for name in ('dynamic_dropout','observation_dropout','visual_predictor_dropout'):
        number(c[name], 'model.'+name, minimum=0)
        if c[name] >= 1:
            raise ValueError(name+' must be less than 1')
    number(c['observation_initial_gate'], 'model.observation_initial_gate', positive=True)
    if c['observation_initial_gate'] >= 1:
        raise ValueError('observation_initial_gate must be less than 1')
    number(c['encoder_interpolate_offset'], 'model.encoder_interpolate_offset', minimum=0)
    boolean(c['encoder_interpolate_antialias'], 'model.encoder_interpolate_antialias')
    lo = c['lora_config']
    keys(lo, {'rank','alpha','dropout','target_patterns','target_blocks','adapt_conv1x1'}, 'model.lora_config')
    integer(lo['rank'], 'lora.rank'); number(lo['alpha'], 'lora.alpha', positive=True)
    number(lo['dropout'], 'lora.dropout', minimum=0)
    if lo['dropout'] >= 1:
        raise ValueError('lora.dropout must be less than 1')
    if not isinstance(lo['target_patterns'], list) or not lo['target_patterns'] or not all(isinstance(s,str) and s for s in lo['target_patterns']):
        raise ValueError('lora.target_patterns must be a nonempty string list')
    if lo['target_blocks'] is not None:
        if not isinstance(lo['target_blocks'], list) or not lo['target_blocks']:
            raise ValueError('lora.target_blocks must be a nonempty list or null')
        for block in lo['target_blocks']:
            integer(block, 'lora.target_blocks', 0)
            if block >= 12:
                raise ValueError('ViT-S/14 has 12 blocks')
    boolean(lo['adapt_conv1x1'], 'lora.adapt_conv1x1')

def model_config(config=None):
    value = load_world_model()['model'] if config is None else deepcopy(config.get('model', config))
    # Derived internal fields are accepted only with the correct values.
    derived = {k:value.pop(k) for k in list(value) if k in ('visual_state_dim','visual_token_count','coordinate_classes','coordinate_soft_sigma')}
    _validate_model(value)
    value.update(visual_state_dim=384, visual_token_count=(value['image_size']//value['encoder_patch_size'])**2,
                 coordinate_classes=2*value['coordinate_limit']+2, coordinate_soft_sigma=1.0)
    for name, old in derived.items():
        if value[name] != old:
            raise ValueError(f'{name} is derived and cannot be set independently')
    return value

def load_navigation(path=DEFAULT_DIR/'navigation.yaml'):
    return navigation_config(read_yaml(path))

def navigation_config(doc):
    doc = deepcopy(doc)
    keys(doc, NAV_KEYS, 'navigation')
    for name in ('episode_limit','candidates','horizon','execute_steps','max_batches'):
        integer(doc[name], 'navigation.'+name)
    for name in ('branch_seed','execution_seed'):
        integer(doc[name], 'navigation.'+name, 0)
    if doc['candidates'] < 8 or doc['horizon'] > 62 or doc['execute_steps'] > doc['horizon']:
        raise ValueError('Require candidates>=8, horizon<=62 (collision bitmask), execute_steps<=horizon')
    sigma = doc['sigma_degrees']
    if not isinstance(sigma, list) or len(sigma) != doc['horizon'] or sigma[0] is not None:
        raise ValueError('sigma_degrees must match horizon and start with null (uniform first step)')
    for value in sigma[1:]: number(value, 'sigma_degrees', positive=True)
    number(doc['reachability_epsilon'], 'reachability_epsilon', minimum=0, maximum=1)
    number(doc['temperature'], 'temperature', positive=True)
    number(doc['arrival_reward'], 'arrival_reward', minimum=0)
    field = doc['field']; keys(field, FIELD_KEYS, 'navigation.field')
    for name in ('rho','half_life','goal_gate_length','repulsion_radius'): number(field[name], 'field.'+name, positive=True)
    for name in ('goal_strength','repel_strength'): number(field[name], 'field.'+name, minimum=0)
    boolean(field['initial_repulsion'], 'field.initial_repulsion')
    if field['capacity'] is not None: integer(field['capacity'], 'field.capacity')
    integer(field['registration_interval'], 'field.registration_interval', 0)
    reg=doc['registration']; keys(reg, REG_KEYS, 'navigation.registration')
    integer(reg['window'], 'registration.window', 2); integer(reg['half_window'], 'registration.half_window')
    if reg['half_window']*2 != reg['window']:
        raise ValueError('registration.window must equal 2*half_window')
    number(reg['epsilon'], 'registration.epsilon', minimum=0)
    number(reg['distance_improvement'], 'registration.distance_improvement', minimum=0)
    number(reg['repeat_fraction'], 'registration.repeat_fraction', minimum=0, maximum=1)
    for name in ('history_enabled','no_gain_enabled'): boolean(reg[name], 'registration.'+name)
    keys(doc['guard'], {'enabled','replan_on_new_passage'}, 'navigation.guard')
    keys(doc['execution'], {'replan_on_prediction_mismatch'}, 'navigation.execution')
    for group in ('guard','execution'):
        for name, value in doc[group].items(): boolean(value, group+'.'+name)
    doc.update(prior_candidates=doc['candidates']-8, coverage_candidates=8, gamma=1.0,
               reward=dict(success=doc['arrival_reward'],progress=0.0,first_visit=0.0,collision_first=0.0,
                           collision_base=1.0,repeat_coefficient=0.0,step_cost=0.0))
    field.update(deduplication='exact_coordinate_keep_birth', repulsion_kernel='compact_quartic')
    return doc

def load_world(path=DEFAULT_DIR/'world.yaml'):
    path=Path(path).expanduser().resolve();doc=read_yaml(path)
    keys(doc, {'source','map_path','seed','geometry','generation','environment','task'}, 'world')
    if doc['source'] not in ('generated','file'): raise ValueError('world.source must be generated or file')
    doc['map_path']=_asset(doc['map_path'], path.parent, 'world.map_path')
    if doc['source']=='file' and doc['map_path'] is None: raise ValueError('File world requires map_path')
    integer(doc['seed'], 'world.seed', 0)
    keys(doc['geometry'], GEO_KEYS, 'world.geometry');keys(doc['generation'], GEN_KEYS, 'world.generation')
    keys(doc['environment'], {'max_action_norm','collision_substep'}, 'world.environment')
    keys(doc['task'], {'start','goal','coordinate_mode'}, 'world.task')
    for name,value in doc['geometry'].items(): integer(value, 'geometry.'+name)
    for name,value in doc['generation'].items(): integer(value, 'generation.'+name, 0 if name=='room_clearance' else 1)
    for name,value in doc['environment'].items(): number(value,'environment.'+name,positive=True)
    if doc['task']['coordinate_mode'] not in ('absolute_patch','relative_patch'):
        raise ValueError('task.coordinate_mode must be absolute_patch or relative_patch')
    for name in ('start','goal'):
        value=doc['task'][name]
        if value is not None and (not isinstance(value,list) or len(value)!=2 or not all(type(x) is int for x in value)):
            raise ValueError('task.'+name+' must be two integer patch coordinates or null')
    validate_geometry({**doc['geometry'], **doc['generation'], **doc['environment']})
    grid=doc['geometry']['map_tiles_per_axis']*doc['geometry']['tile_size']//doc['geometry']['patch_size']
    for name in ('start','goal'):
        value=doc['task'][name]
        absolute = name=='start' or doc['task']['coordinate_mode']=='absolute_patch'
        if value is not None and absolute and any(not 0 <= x < grid for x in value):
            raise ValueError('task.'+name+' is outside the world grid')
    return doc

def validate_geometry(c):
    for name in GEO_KEYS | GEN_KEYS:
        integer(c[name], 'world.'+name, 0 if name=='room_clearance' else 1)
    for name in ('max_action_norm','collision_substep'):
        number(c[name], 'world.'+name, positive=True)
    if c['viewport_size']%2 or c['patch_size']%2 or c['ball_size']%2:
        raise ValueError('Viewport, patch and square-agent sizes must be even pixel counts')
    size=c['map_tiles_per_axis']*c['tile_size'];patch=c['patch_size']
    for name in ('tile_size','wall_thickness','ball_size','room_clearance','min_room_width','max_room_width','min_room_height','max_room_height','door_width'):
        if c[name]%patch:
            raise ValueError(f'world.{name} must align to patch_size')
    if c['door_width']%(2*patch) or c['door_width'] <= c['ball_size']:
        raise ValueError('door_width must exceed ball_size and be divisible by 2*patch_size')
    for low,high in (('min_room_count','max_room_count'),('min_doors_per_room','max_doors_per_room'),('min_room_width','max_room_width'),('min_room_height','max_room_height')):
        if c[low]>c[high]: raise ValueError(f'{low} must not exceed {high}')
    for axis in ('width','height'):
        if c['min_room_'+axis] < 2*c['wall_thickness']+c['door_width']+2:
            raise ValueError('Room '+axis+' is too small for walls and doors')
        if c['max_room_'+axis] > size-2*(c['wall_thickness']+c['room_clearance']):
            raise ValueError('Room '+axis+' cannot fit inside the map margins')

def validate_world_model(world, model):
    g=world['geometry'];size=g['map_tiles_per_axis']*g['tile_size']
    if g['viewport_size'] != model['image_size']:
        raise ValueError('world.viewport_size must equal model.image_size')
    if size % g['patch_size'] or size//g['patch_size']-1 > model['coordinate_limit']:
        raise ValueError('World grid exceeds model.coordinate_limit or is not divisible by patch_size')
    goal=world['task']['goal']
    if goal is not None and world['task']['coordinate_mode']=='relative_patch' and any(abs(x)>model['coordinate_limit'] for x in goal):
        raise ValueError('Relative task goal exceeds model.coordinate_limit')

def load_agent_config(directory=DEFAULT_DIR):
    directory=Path(directory).expanduser().resolve()
    model=load_world_model(directory/'world_model.yaml');nav_raw=read_yaml(directory/'navigation.yaml')
    nav=navigation_config(nav_raw);world=load_world(directory/'world.yaml')
    validate_world_model(world,model['model'])
    return {'world_model':model,'navigation':nav_raw,'world':world},nav

def save_resolved_config(directory, documents):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name,document in documents.items():
        (directory/(name+'.yaml')).write_text(yaml.safe_dump(document,sort_keys=False,allow_unicode=True))
