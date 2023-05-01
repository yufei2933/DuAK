# -*- coding: UTF-8 -*-
import json
from BFS.KB import KB
from BFS.BFS import BFS

from collections import namedtuple, Counter
from utils import *
from config import *


def path_clean(path):
    rel_ents = path.split(' -> ')
    relations = []
    entities = []
    for idx, item in enumerate(rel_ents):
        if idx % 2 == 0:
            relations.append(item)
        else:
            entities.append(item)
    entity_stats = Counter(entities).items()
    duplicate_ents = [item for item in entity_stats if item[1] != 1]
    duplicate_ents.sort(key=lambda x: x[1], reverse=True)
    for item in duplicate_ents:
        ent = item[0]
        ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
        if len(ent_idx) != 0:
            min_idx = min(ent_idx)
            max_idx = max(ent_idx)
            if min_idx != max_idx:
                rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
    return ' -> '.join(rel_ents)


def prob_norm(probs):
    return probs / sum(probs)


import paddle


def max_tensor_index(tensor):
    i = 0
    index = []
    max_value = paddle.max(tensor)
    for value in tensor[0]:
        if value == max_value:
            index.append(i)
        i += 1
    return index


def teacher(e1, e2, num_paths, env, path=None):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))
    f = open(path)
    content = f.readlines()
    f.close()
    kb = KB()
    for line in content:
        ent1, rel, ent2 = line.rsplit()
        kb.addRelation(ent1, rel, ent2)
    # kb.removePath(e1, e2)
    intermediates = kb.pickRandomIntermediatesBetween(e1, e2, num_paths)
    res_entity_lists = []
    res_path_lists = []
    for i in range(num_paths):
        suc1, entity_list1, path_list1 = BFS(kb, e1, intermediates[i])
        suc2, entity_list2, path_list2 = BFS(kb, intermediates[i], e2)
        if suc1 and suc2:
            res_entity_lists.append(entity_list1 + entity_list2[1:])
            res_path_lists.append(path_list1 + path_list2)
    print('BFS found paths:', len(res_path_lists))

    # ---------- clean the path --------
    res_entity_lists_new = []
    res_path_lists_new = []
    for entities, relations in zip(res_entity_lists, res_path_lists):
        rel_ents = []
        for i in range(len(entities) + len(relations)):
            if i % 2 == 0:
                rel_ents.append(entities[int(i / 2)])
            else:
                rel_ents.append(relations[int(i / 2)])

        # print(rel_ents)

        entity_stats = Counter(entities).items()
        duplicate_ents = [item for item in entity_stats if item[1] != 1]
        duplicate_ents.sort(key=lambda x: x[1], reverse=True)
        for item in duplicate_ents:
            ent = item[0]
            ent_idx = [i for i, x in enumerate(rel_ents) if x == ent]
            if len(ent_idx) != 0:
                min_idx = min(ent_idx)
                max_idx = max(ent_idx)
                if min_idx != max_idx:
                    rel_ents = rel_ents[:min_idx] + rel_ents[max_idx:]
        entities_new = []
        relations_new = []
        for idx, item in enumerate(rel_ents):
            if idx % 2 == 0:
                entities_new.append(item)
            else:
                relations_new.append(item)
        res_entity_lists_new.append(entities_new)
        res_path_lists_new.append(relations_new)

    # print(res_entity_lists_new)
    # print(res_path_lists_new)

    good_episodes = []
    targetID = env.entity2id_[e2]
    for path in zip(res_entity_lists_new, res_path_lists_new):
        good_episode = []
        for i in range(len(path[0]) - 1):
            currID = env.entity2id_[path[0][i]]
            nextID = env.entity2id_[path[0][i + 1]]
            state_curr = [currID, targetID, 0]
            state_next = [nextID, targetID, 0]
            actionID = env.relation2id_[path[1][i]]
            good_episode.append(
                Transition(state=env.idx_state(state_curr), action=actionID, next_state=env.idx_state(state_next),
                           reward=1))
        good_episodes.append(good_episode)
    return good_episodes


def duplicate_removal(list_tuple):
    result = []
    for i in list_tuple:
        try:
            if i not in result:  # 有tensor,不支持in操作，可能？
                result.append(i)
        except:
            if type(i) == 'paddle.Tensor':
                i = paddle.tolist(i)
                result.append(paddle.to_tensor(i))
    return result


def parameter_sharing(pos_episode, neg_episode):
    path_relation_found = []
    # if find similar entity, which represents a whole episode have been found.
    pos_observations = []
    neg_observations = []
    pos_states=[]
    neg_states=[]
    pos_actions=[]
    neg_actions=[]
    #print("pos_episode",pos_episode)
    pos_count=0
    neg_count=0
    
    for observation in pos_episode["next_obs"]:
        if observation!=None:
            pos_observations.append(observation)
            pos_states.append(pos_episode["states"][pos_count])
            pos_actions.append(pos_episode["actions"][pos_count])
        pos_count+=1
    for observation in neg_episode["next_obs"]:
        if observation!=None:
            neg_observations.append(observation)
            neg_states.append(neg_episode["states"][neg_count])
            neg_actions.append(neg_episode["actions"][neg_count])
        neg_count+=1
    pos_observations_re=list(set(pos_observations))
    neg_observations_re=list(set(neg_observations))
    path_episodes_pos=[]
    path_episodes_neg=[]
    path_transition={"states":[],"actions":[],"next_states":[],"rewards":[],"next_obs":[]}
    if len(set(pos_observations_re + neg_observations_re)) < len(pos_observations_re + neg_observations_re):
        same_obs = set(pos_observations_re) & set(neg_observations_re) 
        for obs in same_obs:  
            index_pos=pos_observations.index(obs)
            index_neg=pos_observations.index(obs)
            if index_neg or index_pos==0:
                continue
            state_pos=pos_states[:index_pos+1]+neg_states[1:index_neg][::-1]
            state_neg=neg_states[:index_neg+1]+pos_states[1:index_pos][::-1]
        
            action_pos=pos_actions[:index_pos]+neg_actions[:index_neg][::-1]
            action_neg=neg_actions[:index_neg]+pos_actions[:index_pos][::-1]

            path_transition["states"]=state_pos
            path_transition["actions"]=action_pos
            path_transition["rewards"].append(0.5*index_pos/(index_pos+index_neg))
            path_episodes_pos.append(path_transition)
            with open(TASK_RESULT_PATH+"sharing_pos_path.txt","a") as f1:
                f1.write(str(action_pos)+"\n")
            path_transition["states"]=state_neg
            path_transition["actions"]=action_neg
            path_transition["rewards"].append(0.5*index_neg/(index_pos+index_neg))
            path_episodes_neg.append(path_transition)
            with open(TASK_RESULT_PATH+"sharing_neg_path.txt","a") as f2:
                f2.write(str(action_pos)+"\n")
    else:
        path_episodes_pos=None
        path_episodes_neg=None
    return path_episodes_pos,path_episodes_neg

def sharing_eposide_path_extract(episodes,env):
    path_only_relation = []
    i=0
    for each_episode in episodes:
        temp_episode=[]
        print(each_episode)
        for scence in each_episode:
            temp_episode.append(env.relations[scence.action])
        path_only_relation.append(" -> ".join(temp_episode))
        i+=1

    return path_only_relation

def paddle_to_tensor(npnarry, dtype="float32"):
    # print("\n",type(npnarry))
    try:
        if list(npnarry):
            return paddle.to_tensor(npnarry, dtype=dtype)
    except:

        return npnarry


import numpy as np


def trans_txt_to_ndarray(filename):
    entity = {}
    with open(filename, "r") as f1:
        content = f1.readlines()
        for line in content:
            id_vec = line.split("\t")
            id = id_vec[0]
            str_ = id_vec[1]
            list_ = str_.strip()
            list_ = list_.strip("[")
            list_ = list_.strip("]")
            list_ = list_.split(",")
            list_ = list(map(float, list_))
            vec = np.array(list_).astype("float32")
            # print(vec.shape, type(vec), vec)
            entity[int(id)] = vec
    return entity
def trans_txt_to_ndarry_dict(filename):
    content_dict={}
    content_ndarry=np.loadtxt(filename)
    count=0
    for entity in content_ndarry:
        content_dict[count]=content_ndarry[count,:]
        count+=1
    return content_dict

def cosine_distance(vec1, vec2):
    vec1 = paddle_to_tensor(vec1, dtype="float32")
    vec2 = paddle_to_tensor(vec2, dtype="float32")
    point1 = vec1 / paddle.norm(vec1, axis=0, keepdim=True)  
    point2 = vec2 / paddle.norm(vec2, axis=0, keepdim=True)
    CosineSimilarities = paddle.sum(paddle.multiply(point1, point2), axis=-1)

    return CosineSimilarities
