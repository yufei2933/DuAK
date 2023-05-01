# -*- coding: UTF-8 -*-
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optimizer
from paddle.distribution import Categorical
import numpy as np
from config import *
from networks import Policy_nn
paddle.device.set_device(GPU_SET)

class PG:
    def __init__(self,state_dim,action_dim):
        self.gamma=0.99
        self.policy=Policy_nn(state_dim,action_dim)
        self.optimizer=optimizer.Adam(learning_rate=0.001, parameters=self.policy.parameters())
        self.action_dim=action_dim

    def sample(self,state,experience_prob=None):
        state=paddle.to_tensor(state,dtype="float32")
        action_prob=self.policy(state)
        if type(experience_prob)=='paddle.Tensor':
            action_prob=experience_prob*action_prob
        action = np.random.choice(range(self.action_dim), p=action_prob.cpu().numpy())
        return action

    def get_ut(self,reward_list,gamma=1.0):
        for i in range(len(reward_list)-2,-1,-1):
            reward_list[i]+=gamma*reward_list[i+1]
        return np.array(reward_list)

    def learn(self,transition_dict):
        state=paddle.to_tensor(transition_dict["states"],dtype="float32")
        action=paddle.to_tensor(transition_dict["actions"],dtype="int32")
        reward=paddle.to_tensor(self.get_ut(transition_dict["rewards"],self.gamma),dtype="float32")
        act_prob=self.policy(state)
        log_prob=paddle.sum(-1.0*paddle.log(act_prob)*paddle.nn.functional.one_hot(action, action_space),axis=-1)
        self.optimizer.clear_grad()
        loss=log_prob*reward
        loss=sum(loss)#mean(loss)
        loss.backward()
        self.optimizer.step()
        return loss
    def learn_state_action(self,state,action,reward=1.0):
        print(reward)
        state=paddle.to_tensor(state,dtype="float32")
        action=paddle.to_tensor(action,dtype="int32")
        act_prob=self.policy(state)
        #print(act_prob)
        log_prob=paddle.sum(-1.0*paddle.log(act_prob)*paddle.nn.functional.one_hot(action, action_space),axis=-1)
        self.optimizer.clear_grad()
        loss=log_prob*reward
        loss=sum(loss)#mean(loss)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def save_model(self,path):
        paddle.save(self.policy.state_dict(),path=path)
    def load_model(self,path):
        layer_state_dict = paddle.load(path)
        self.policy.set_state_dict(layer_state_dict)