from agents.nn.hyper_network import Hyper_Network
import torch.nn as nn

class Hyper_Critic(nn.Module):

    def __init__(self, state_dim, action_dim, h_dim):
        super(Hyper_Critic).__init__()

        meta_v_dim = state_dim
        base_v_dim = action_dim
        self.q1 = Hyper_Network(meta_v_dim, base_v_dim, 1, h_dim)
        self.q2 = Hyper_Network(meta_v_dim, base_v_dim, 1, h_dim)

    def forward(self, state, action, debug=None):
        q1 = self.q1(state, action, debug)
        q2 = self.q2(state, action, debug)
        return q1, q2

    def Q1(self, state, action, debug):
        q1 = self.q1(state, action, debug)
        return q1