from agents.nn.hyper_network import Hyper_Network
import torch.nn as nn

class Hyper_Policy(nn.Module):

    def __init__(self, task_dim, state_dim, action_dim, h_dim, max_action):
        super(Hyper_Policy, self).__init__()

        self.task_dim = task_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hidden = h_dim
        self.max_action = max_action
        self.hyper = Hyper_Network(meta_v_dim=task_dim, base_v_dim=state_dim, output_dim=action_dim, h_dim=h_dim)

    def forward(self, state, task):
        z = self.hyper.get_z(task)
        action = self.hyper.dynamic(z, state)
        return self.max_action * torch.tanh(action)