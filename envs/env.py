# -*- coding: UTF-8 -*-
import numpy as np
import random
import json
import paddle

from utils import *
from config import *
paddle.device.set_device(GPU_SET)
"""
目录data_analysis增加__init__.py文件会强行将该目录层次提高一层。
"""

class Env(object):

    def __init__(self,task=None):
        with open(entities_id) as f1:
            self.entity2id_=json.load(f1)
        with open(relations_id) as f2:
            self.relation2id_=json.load(f2)
        if dataset_name=="NELL-995":
            self.entity2vec = trans_txt_to_ndarry_dict(DATASET + 'entity2vec.bern')
            self.relation2vec = trans_txt_to_ndarry_dict(DATASET + 'relation2vec.bern')
        else:
            self.entity2vec=trans_txt_to_ndarray(entity_dim)
            self.relation2vec =trans_txt_to_ndarray(relation_dim)

        self.relations = []
        for k,v in self.relation2id_.items():
            self.relations.append(k)

        self.path = []
        self.path_relations = []

        f = open(KB_ENV_RL)
        kb_all = f.readlines()
        f.close()
        self.kb = []
        if task != None:
            relation = task
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation + '_inv':
                    self.kb.append(line)
        self.die = 0
    def reset(self):
        done=False
        return done

    def step(self, state, action):

        done = False  # Whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]
        choices = []

        for line in self.kb:
            triple = line.rsplit()#line.split("\t")#h,t,r格式
            e1_idx = self.entity2id_[triple[0]]#head-entity

            if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
                choices.append(triple)

        penalty_reward=-1/max_steps
        
        if len(choices) == 0:#没有有效路径被发现，主要受action的影响。
            reward = -1+penalty_reward
            self.die += 1
            next_state = state  #stay in the initial state
            next_state[-1] = self.die#
            new_obs=None
            return (reward, next_state, new_obs,done)

        else:# find a valid step
            path = random.choice(choices)#
            self.path.append(path[2] + ' -> ' + path[1])#
            self.path_relations.append(path[2])#
            self.die = 0
            new_obs=path[1]
            new_pos = self.entity2id_[path[1]]#
            new_state = [new_pos, target_pos, self.die]#

            old_cosine_reward = cosine_distance(self.entity2vec[state[0]], self.entity2vec[state[1]])
            new_cosine_reward = cosine_distance(self.entity2vec[new_pos], self.entity2vec[state[1]])

            if new_cosine_reward > old_cosine_reward:
                cosine_reward = +0.05
            else:
                cosine_reward = -0.005

            reward = penalty_reward + cosine_reward
            if new_pos == target_pos:
                #print('Find a path:', self.path)
                done = True
                
                new_state = None
                new_obs = None
                path_length = len(self.path)
                length_reward = 1 / path_length
                global_reward = 1
                
                reward =global_reward +length_reward+penalty_reward
            return (reward, new_state, new_obs, done)

    def idx_state(self,idx_list):#input: [env.entity2id_[entity_head], env.entity2id_[entity_tail], 0]
        #print("idx_state",idx_list)
        if idx_list != None:
            curr = self.entity2vec[idx_list[0]]
            targ = self.entity2vec[idx_list[1]]
            return np.concatenate((curr, targ - curr))  # concatenate 拼接列表
        else:
            return None

    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id_[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id_[triple[2]])
        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self.relation2vec[self.relation2id_[relation], :] for relation in path]
        embeddings = np.reshape(embeddings, (-1, embedding_dim))
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, (-1, embedding_dim))

    



