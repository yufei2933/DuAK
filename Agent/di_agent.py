# -*- coding: UTF-8 -*-
from itertools import count
import collections

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical
import time

from Agent.networks import Policy_nn
from Agent.agent import PG
from utils import *
from config import *
if dataset_name=="FB15K-237":
    from data_analysis.entity_relation_fb import Analyse
elif dataset_name=="NELL-995":
    from data_analysis.entity_relation_nell import Analyse
elif dataset_name=="CRD":
    from data_analysis.entity_relation_crd import Analyse
    
paddle.device.set_device(GPU_SET)
class BaseAgent:
    def __init__(self,name):
        self.name=name
        self.success = False
        #double agents
        self.agent=PG(state_dim, action_space)
        self.path_found_entity=[]
        with open(relations_id,"r") as f2:
            relation2id_=json.load(f2)
        
        self.history_analyse=Analyse(relation2id_)
    def explore(self,env,entity_head,entity_tail):
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_obs', 'reward'))

        temp_entity_class_count, temp_counts_tensor_norm = self.history_analyse.class_analyse(entity_head)  # tensor格式
        state_idx = [env.entity2id_[entity_head], env.entity2id_[entity_tail], 0]
        
        Episode_transition={"states":[],"actions":[],"next_states":[],"rewards":[],"next_obs":[]}
        state_batch_negative = []
        action_batch_negative = []
        reward_batch_negative = []
        print("3. Starting to reforcement learning... ...")
        #######################################################
        obs_cur=entity_head
        for t in count():
            # 获取当前state
            state = paddle.to_tensor(env.idx_state(state_idx), dtype="float32")  #200dim state_idx【】
            #print("state",state.shape,"\t",state)
            # 策略网络获取action选择分布
            action= self.agent.sample(state,temp_counts_tensor_norm)#
            
            reward, next_state, new_obs, done = env.step(state_idx, action)
            #print("--Agent interaction--")
            if reward == -1:  #the action fails for this step, means that an failed episode.
                state_batch_negative.append(state.cpu().numpy())
                action_batch_negative.append(action)
                reward_batch_negative.append(reward)
            
            state_idx = next_state
            next_state = env.idx_state(next_state)  # 下一次交互开始
            # episode把此次交互加入
            Episode_transition['states'].append(state)
            Episode_transition['actions'].append(action)
            Episode_transition['next_states'].append(next_state)
            Episode_transition['rewards'].append(reward)
            Episode_transition['next_obs'].append(obs_cur)
            if done or t == max_steps:
                break
            if new_obs:
                self.history_analyse.experience_pool_update(action,obs_cur)
                temp_entity_class_count, temp_counts_tensor_norm = self.history_analyse.class_analyse(new_obs)
                obs_cur=new_obs
        return done,Episode_transition

    
    def run(self,env,entity_head,entity_tail):

        done,epi_transition=self.explore(env,entity_head,entity_tail)
        path_found_entity=[]
        path_relation_found=[]
        if done:
            path_found_entity.append(path_clean(' -> '.join(env.path)))#
            self.agent.learn(epi_transition)
        else:
            try:
                good_episodes = teacher(entity_head, entity_tail, 1, env, GRAPH_PATH)
                for item in good_episodes:
                    teacher_state_batch = []
                    teacher_action_batch = []
                    rewawrd_action_batch = []
                    total_reward = 1 + 1 * 1 / len(item)
                    for t, transition in enumerate(item):
                        teacher_state_batch.append(transition.state)
                        teacher_action_batch.append(transition.action)
                        rewawrd_action_batch.append(transition.reward)
                    self.agent.learn_state_action(teacher_state_batch, teacher_action_batch,total_reward)
            except Exception as e:
                print('Teacher guideline failed')
        
        for path in path_found_entity:
            rel_ent = path.split(' -> ')
            path_relation = []
            for idx, item in enumerate(rel_ent):
                if idx % 2 == 0:
                    path_relation.append(item)
            path_relation_found.append(' -> '.join(path_relation))
        relation_path_stats = collections.Counter(path_relation_found).items()
        relation_path_stats = sorted(relation_path_stats, key=lambda x: x[1], reverse=True)
        with open(TASK_RESULT_PATH +'path_states_'+self.name+'.txt', 'a') as f:
            for item in relation_path_stats:
                f.write(item[0] + '\t' + str(item[1]) + '\n')
        return done,epi_transition
    
    def save_model(self,name,epoch):
        path=AGENT_TRAIN_PATH+name+str(epoch)+".params"
        self.agent.save_model(path)
    def load_model(self,epoch,name):
        path=AGENT_TRAIN_PATH+name+str(epoch)+".params"
        self.agent.load_model(path)
    def save_pre_model(self,name):
        path=PRE_TRAIN_PATH+name+"_pre_train.params"
        self.agent.save_model(path)
    def load_pre_model(self,name):
        path=PRE_TRAIN_PATH+name+"_pre_train.params"
        self.agent.load_model(path)