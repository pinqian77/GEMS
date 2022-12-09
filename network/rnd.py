import torch
import torch.nn as nn

from network.MHAEncoder import MHAEncoder

class RND(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RND, self).__init__()

        self.target_mha = MHAEncoder()
        self.target_out = nn.Linear(state_dim + action_dim, 512)

        self.predictor_mha = MHAEncoder()
        self.predictor_out = nn.Linear(state_dim + action_dim, 512)

        for param in self.target_mha.parameters():
            param.requires_grad = False
        for param in self.target_out.parameters():
            param.requires_grad = False

    def forward(self, state, action):
        # target
        target_feature, enc_self_attns = self.target_mha(state)
        target_feature = target_feature.reshape(target_feature.shape[0], -1)
        target_feature = torch.cat((target_feature, action), 1)
        target_feature = self.target_out(target_feature)

        # predictor
        predict_feature, enc_self_attns = self.predictor_mha(state)
        predict_feature = predict_feature.reshape(predict_feature.shape[0], -1)
        predict_feature = torch.cat((predict_feature, action), 1)
        predict_feature = self.target_out(predict_feature)

        return predict_feature, target_feature