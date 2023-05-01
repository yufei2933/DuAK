# -*- coding: UTF-8 -*-

import paddle.nn as nn
import paddle
import paddle.nn.functional as F
from paddle.distribution import Categorical
import numpy as np

class Policy_nn_cosine(nn.Layer):
    def __init__(self, state_size, action_size, ):
        super(Policy_nn_cosine, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = paddle.nn.Linear(self.state_size, 512)
        self.linear2 = paddle.nn.Linear(512, 1024)
        self.linear3 = paddle.nn.Linear(1024, self.action_size)
    def forward(self,state,):
        linear1 = F.relu(self.linear1(state))
        linear2 = F.relu(self.linear2(linear1))
        out = self.linear3(linear2)
        return out
    def select_action(self, state, temp_counts_tensor_norm):
        out = self.forward(state)
        new_prob = paddle.multiply(temp_counts_tensor_norm, out)
        net_out = F.softmax(new_prob, axis=-1)
        
        distribution = Categorical(distribution)
        new_prob.stop_gradient = True
        net_out=F.softmax(out)
        return net_out, distribution

    def update(self, rewards, log_probs):
        loss = 0
        for log_prob, reward in zip(rewards, log_probs):
            action_loss = -log_prob * reward
            loss += action_loss
        return loss

    def update_org(self, states, actions, global_reward):
        states = paddle.to_tensor(states)
        out = self.forward(states)
        loss = []
        for i in range(len(actions)):
            action_loss = -paddle.log(out[i][0][actions[i]]) * global_reward
            loss.append(action_loss)
        return loss

class Policy_nn(nn.Layer):
    def __init__(self, state_size, action_size, ):
        super(Policy_nn, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = paddle.nn.Linear(self.state_size, 512)
        self.linear2 = paddle.nn.Linear(512, 1024)
        self.linear3 = paddle.nn.Linear(1024, self.action_size)
    def forward(self,state):
        linear1 = F.sigmoid(self.linear1(state))
        linear2 = F.sigmoid(self.linear2(linear1))
        linear3 = self.linear3(linear2)
        out=F.softmax(linear3,axis=-1)
        return out
    def select_action(self, state, temp_counts_tensor_norm):
        out=self.forward(state)
        new_prob = paddle.multiply(temp_counts_tensor_norm, out)
        distribution=F.softmax(new_prob, axis=-1)
        distribution = Categorical(distribution)
        new_prob.stop_gradient = True
        return distribution
