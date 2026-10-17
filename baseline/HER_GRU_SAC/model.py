"""Only the actor network used by EXP141; parameter names match actor.*."""
import torch
from torch import nn
from torch.nn import functional as F


AUX_DIM = 19


def head(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 8))


class ObservationEncoder(nn.Module):
    def __init__(self, state_input_dim, state_embedding_size):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(384, 16), nn.LayerNorm(16), nn.SiLU())
        self.spatial = nn.Sequential(nn.Linear(4096, 256), nn.LayerNorm(256), nn.SiLU())
        self.state = nn.Sequential(nn.Linear(state_input_dim, state_embedding_size), nn.ReLU())
        self.output_size = 256 + state_embedding_size

    def forward(self, visual, state):
        visual_embedding = self.spatial(self.token(visual.float()).flatten(-2))
        return torch.cat((visual_embedding, self.state(state.float())), dim=-1)


class RecurrentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.observation = ObservationEncoder(AUX_DIM, config.state_embedding_size)
        self.action = nn.Sequential(nn.Linear(8, config.action_embedding_size), nn.ReLU())
        self.reward = nn.Sequential(nn.Linear(1, config.reward_embedding_size), nn.ReLU())
        input_size = self.observation.output_size + config.action_embedding_size + config.reward_embedding_size
        self.gru = nn.GRU(input_size, config.rnn_hidden_size, batch_first=True)
        # Every tensor is strictly loaded from the actor checkpoint before use.
        self.output_size = self.observation.output_size + config.rnn_hidden_size

    def forward(self, visual, previous_actions, received_rewards, state, hidden=None):
        current_observation = self.observation(visual, state)
        previous = F.one_hot(previous_actions.clamp_min(0), 8).float()
        previous *= (previous_actions >= 0).unsqueeze(-1)
        action_embedding = self.action(previous)
        reward_embedding = self.reward(received_rewards.float().unsqueeze(-1))
        history, hidden = self.gru(
            torch.cat((action_embedding, reward_embedding, current_observation), dim=-1), hidden)
        return history, current_observation, hidden


class RecurrentActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recurrent = RecurrentEncoder(config)
        self.policy = head(self.recurrent.output_size)

    def forward(self, visual, previous_actions, received_rewards, state, hidden=None):
        history, current_observation, hidden = self.recurrent(
            visual, previous_actions, received_rewards, state, hidden)
        logits = self.policy(torch.cat((history, current_observation), dim=-1))
        return logits, hidden, history
