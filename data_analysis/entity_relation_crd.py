
# -*- coding: UTF-8 -*-
import paddle
import numpy as np
import time
from config import *
paddle.device.set_device(GPU_SET)
relation_length = 20

class Analyse:
    def __init__(self,relation_to_id):
        self.entity_classes_count=self.get_entity_relation()
        self.relation2id=relation_to_id
        self.entity_class_experience={}#class1:[1,2,3,3,4,,3,3,3,2]
        self.class_experience()

    def all_entity_classes(self,entity_class,entity):

        if entity_class in self.entity_classes:
            if entity in self.entity_classes[entity_class]:
                pass
            else:
                self.entity_classes[entity_class].append(entity)
        else:
            self.entity_classes[entity_class]=[entity]

    def get_entity_relation(self):
        self.entity_classes = {}
        with open(KB_ENV_RL,"r",encoding="utf-8") as f:#正常应该为 ../Dataset/nell/train_tasks.json
            content=f.readlines()
        #concept_televisionstation_wqpt_tv	concept_company_pbs	concept:agentbelongstoorganization
        entity_classes_count = {}

        for triple in content:
            temp_triple=triple.strip().split("\t")
            #print(temp_triple)
            relation=temp_triple[2]
            head_entity=temp_triple[0]
            tail_entity=temp_triple[1]
            head_class=head_entity.split("_")[0]
            tail_class=tail_entity.split("_")[0]
            self.all_entity_classes(head_class, head_entity)
            self.all_entity_classes(tail_class, tail_entity)
            if head_class in entity_classes_count:
                if relation in entity_classes_count[head_class]:
                    entity_classes_count[head_class][relation]+=1
                else:
                    entity_classes_count[head_class][relation]=1
            else:
                entity_classes_count[head_class]={}
                entity_classes_count[head_class][relation]=1
        #print(entity_classes_count)

        return entity_classes_count#{class:{relation1:3,relation2:4,}]}

    def class_analyse(self, state="concept_televisionstation_wqpt_tv"):  # concept_televisionstation_wqpt_tv
        for k,v in self.entity_classes.items():
            if state in v:
                temp_entity_class=k
        relation_prob_ = [0.001 for i in range(relation_length)]
        temp_entity_class_count=self.entity_classes_count[temp_entity_class]
        paths=[]
        counts=[]
        for k,v in temp_entity_class_count.items():
            paths.append(k)
            counts.append(v)
        for i in range(len(paths)):
            path_id=self.relation2id[paths[i]]
            relation_prob_[path_id]=counts[i]

        relation_prob_=self.log_softmax(relation_prob_)#

        return temp_entity_class_count,relation_prob_#

    def class_experience(self):
        self.experience_pool={}
        for k,v in self.entity_classes.items():
            temp_entity_class=k
            relation_prob_ = [0.001 for i in range(relation_length)]
            temp_entity_class_count=self.entity_classes_count[temp_entity_class]
            paths = []
            counts = []
            for k, v in temp_entity_class_count.items():
                paths.append(k)
                counts.append(v)
            for i in range(len(paths)):
                path_id = self.relation2id[paths[i]]
                relation_prob_[path_id] = counts[i]
            relation_prob_ = self.log_softmax(relation_prob_)
            self.experience_pool[temp_entity_class]=relation_prob_

    def experience_pool_update(self,action,state="concept_televisionstation_wqpt_tv",):
        for k,v in self.entity_classes.items():
            if state in v:
                temp_entity_class=k
            else:
                pass
        ori_relation_prob=self.experience_pool[temp_entity_class]
        ori_relation_prob=paddle.tolist(ori_relation_prob)
        temp_action_selected_prob=ori_relation_prob[action]
        new_relation_prob=[]
        for i in range(len(ori_relation_prob)):
            if i==action:
                prob=ori_relation_prob[i]+0.01*(1-temp_action_selected_prob)
                new_relation_prob.append(prob)
            else:
                prob=ori_relation_prob[i]-0.01*(1-temp_action_selected_prob)/len(ori_relation_prob)
                new_relation_prob.append(prob)
        new_relation_prob=paddle.to_tensor(new_relation_prob)
        self.experience_pool[temp_entity_class]=new_relation_prob

    def log_softmax(self,counts):
        counts_ = np.log(np.array(counts) + np.ones([len(counts)]))
        counts_log = []
        for i in range(len(counts_)):
            value = counts_[i] / sum(counts_)

            counts_log.append(value)
        return paddle.to_tensor(counts_log)





