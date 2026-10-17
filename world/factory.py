"""Construct an environment from YAML without exposing map truth to Agent."""
from copy import deepcopy
import numpy as np
from .map import LargeRoomWorldConfig, LargeRoomLayoutGenerator, load_layout
from .environment import LargeRoomEnvironment, PatchAgentEnvironment
if "." in __package__:
    from ..util.config import DEFAULT_DIR, load_world, validate_world_model
else:
    from util.config import DEFAULT_DIR, load_world, validate_world_model


def make_world(path=DEFAULT_DIR/'world.yaml', *, model_config=None):
    document = load_world(path)
    if model_config is not None:
        validate_world_model(document, model_config)
    config = LargeRoomWorldConfig(**document['geometry'], **document['generation'], **document['environment'])
    layout = (LargeRoomLayoutGenerator(config).sample(document['seed']) if document['source']=='generated'
              else load_layout(document['map_path']))
    env = PatchAgentEnvironment(LargeRoomEnvironment(layout, config))
    env.default_seed = document['seed']
    env.default_start = document['task']['start']
    env.task = deepcopy(document['task'])
    env.resolved_config = deepcopy(document)
    for name in ('start','goal'):
        position = env.task[name]
        if position is not None and (name=='start' or env.task['coordinate_mode']=='absolute_patch'):
            if not env.environment.is_free(env._to_environment_position(np.asarray(position, dtype=np.int64))):
                raise ValueError(f'Configured task.{name} is blocked')
    return env
