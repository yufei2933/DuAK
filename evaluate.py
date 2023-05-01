import numpy as np

from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense, Activation
import json

from keras import regularizers,optimizers

from BFS.KB import *
from config import *
from utils import *

def get_features():

    with open(relations_id) as f2:
        relation2id=json.load(f2)
    useful_paths = []
    named_paths = []
    f = open(featurePath,"r",encoding="utf-8")
    paths = f.readlines()
    f.close()
    print(len(paths),"found by Agent","\t",paths)

    for line in paths:
        path = line.rstrip()#
        length = len(path.split(' -> '))
        if length <= 10:
            pathIndex = []
            pathName = []
            relations = path.split(' -> ')#==> ['concept:personleadsorganization', 'concept:personleadsorganization_inv', 'concept:personleadsorganization']
            for rel in relations:
                pathName.append(rel)
                rel_id = relation2id[rel]
                pathIndex.append(rel_id)
            useful_paths.append(pathIndex)
            named_paths.append(pathName)
    return useful_paths, named_paths

def train(kb, kb_inv, named_paths):
    f = open(RELATION_TASK_TRAIN_PAIRS,encoding="utf-8")
    train_data = f.readlines()
    f.close()
    train_pairs = []
    train_labels = []
    for line in train_data:
        e1 = line.split(',')[0].replace('thing$', '')
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        train_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        train_labels.append(label)
    training_features = []

    for sample in train_pairs:
        feature = []
        for path in named_paths:
            feature.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
        training_features.append(feature)
    print("training_feature",training_features)
    model = Sequential()
    input_dim = len(named_paths)
    model.add(Dense(1, activation='relu',input_dim=input_dim))
    model.compile(optimizer="rmsprop",loss='binary_crossentropy', metrics=['accuracy'])
    #model.fit(np.array(training_features), np.array(train_labels), nb_epoch=100, batch_size=1024)#nb_epoch error
    model.fit(np.array(training_features), np.array(train_labels), epochs=100, batch_size=1024)
    return model


def evaluate_logic():
    kb = KB()
    kb_inv = KB()

    f = open(GRAPH_PATH,"r",encoding="utf-8")
    kb_lines = f.readlines()
    f.close()

    for line in kb_lines:
        e1 = line.split()[0]
        rel = line.split()[1]
        e2 = line.split()[2]
        kb.addRelation(e1, rel, e2)
        kb_inv.addRelation(e2, rel, e1)

    _, named_paths = get_features()#[['concept:personleadsorganization'], ['concept:organizationhiredperson_inv'],

    model = train(kb, kb_inv, named_paths)

    #单独去测试数据与标签
    f = open(RELATION_TASK_TEST_PAIRS,"r",encoding="utf-8")
    test_data = f.readlines()
    f.close()
    test_pairs = []
    test_labels = []
    # queries = set()
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$', '')
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        test_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        test_labels.append(label)
    print("test_labels",test_labels)
    aps = []
    query = test_pairs[0][0]
    y_true = []
    y_score = []

    score_all = []

    for idx, sample in enumerate(test_pairs):
        if sample[0] == query:
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
            # features = features*path_weights
            score = model.predict(np.reshape(features, [1, -1]))
            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
        else:
            query = sample[0]
            count = sorted(zip(y_score, y_true),reverse=True)  # 上一版本无reverser,因此默认是 升序  reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            ranks = []
            correct = 0
            for idx_, item in enumerate(count):
                if item[1] == 1:#标签为正
                    correct += 1
                    ranks.append(correct / (1.0 + idx_))
            if len(ranks) == 0:
                aps.append(0)
            else:
                aps.append(np.mean(ranks))

            y_true = []
            y_score = []
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
            score = model.predict(np.reshape(features, [1, -1]))
            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])

    count = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    ranks = []
    correct = 0
    for idx_, item in enumerate(count):
        if item[1] == 1:
            correct += 1
            ranks.append(correct / (1.0 + idx_))
    aps.append(np.mean(ranks))
    score_label = zip(score_all, test_labels)
    score_label_ranked = sorted(score_label, key=lambda x: x[0], reverse=True)
    mean_ap = np.mean(aps)
    print('RL MAP: ', mean_ap)
    return mean_ap

def bfs_two(e1,e2,path,kb,kb_inv):
    start = 0
    end = len(path)
    left = set()
    right = set()
    left.add(e1)
    right.add(e2)

    left_path = []
    right_path = []
    while (start < end):
        left_step = path[start]#第一个关系路径
        left_next = set()#到kb中找
        right_step = path[end-1]#最后一个关系路径
        right_next = set()#到kb_inv中找

        if len(left) < len(right):
            left_path.append(left_step)
            start += 1
            for entity in left:
                try:
                    for path_ in kb.getPathsFrom(entity):
                        if path_.relation == left_step:
                            left_next.add(path_.connected_entity)
                except Exception as e:
                    return False
            left = left_next

        else:
            right_path.append(right_step)
            end -= 1
            for entity in right:
                try:
                    for path_ in kb_inv.getPathsFrom(entity):
                        if path_.relation == right_step:
                            right_next.add(path_.connected_entity)
                except Exception as e:
                    return False
            right = right_next

    if len(right & left) != 0:
        return True
    return False

def test_pos_file():
    f=open(RELATION_TASK_TEST_PAIRS,"r")
    test_data = f.readlines()
    f.close()
    entity_pairs_set=[]
    for line in test_data:
        if line[-2] == '+':
            e1 = line.split(',')[0].replace('thing$', '')
            e2 = line.split(',')[1].split(':')[0].replace('thing$', '') 
            entity_pairs=(e1,e2)
            entity_pairs_set.append(entity_pairs)
    entity_pairs_set=list(set(entity_pairs_set))
    with open(RELATION_TASK_TEST_POS,"w") as f1:
        for entity_pairs in entity_pairs_set:
            temp_content=entity_pairs[0]+"\t"+entity_pairs[1]+"\n"
            f1.write(temp_content)

def evaluate_MRR():
    kb = KB()
    kb_inv = KB()

    f = open(GRAPH_PATH,"r",encoding="utf-8")
    kb_lines = f.readlines()
    f.close()

    for line in kb_lines:
        e1 = line.split()[0]
        rel = line.split()[1]
        e2 = line.split()[2]
        kb.addRelation(e1, rel, e2)
        kb_inv.addRelation(e2, rel, e1)

    _, named_paths = get_features()

    model = train(kb, kb_inv, named_paths)
    f = open(RELATION_TASK_TEST_PAIRS,"r",encoding="utf-8")
    test_data = f.readlines()
    f.close()
    test_pairs = []
    test_labels = []
    for line in test_data:
        e1 = line.split(',')[0].replace('thing$', '')
        e2 = line.split(',')[1].split(':')[0].replace('thing$', '')
        if (e1 not in kb.entities) or (e2 not in kb.entities):
            continue
        test_pairs.append((e1, e2))
        label = 1 if line[-2] == '+' else 0
        test_labels.append(label)
    aps = []
    query = test_pairs[0][0]
    y_true = []
    y_score = []

    score_all = []
    ranks = []
    for idx, sample in enumerate(test_pairs):
        if sample[0] == query:
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
            score = model.predict(np.reshape(features, [1, -1]))
            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
        else:
            query = sample[0]
            count = sorted(zip(y_score, y_true),reverse=True)  # reverse = True 降序 ， reverse = False 升序（默认）。
            for idx_, item in enumerate(count):
                if item[1] == 1:
                    ranks.append(1.0 + idx_)
            y_true = []
            y_score = []
            features = []
            for path in named_paths:
                features.append(int(bfs_two(sample[0], sample[1], path, kb, kb_inv)))
            score = model.predict(np.reshape(features, [1, -1]))
            score_all.append(score[0])
            y_score.append(score)
            y_true.append(test_labels[idx])
    count = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    for idx_, item in enumerate(count):
        if item[1] == 1:
            ranks.append(idx_+1)
    rank_rev=[1/seq for seq in ranks]
    mean_rr=1/len(rank_rev)*(sum(rank_rev))
    evaluate_Hit_N(ranks)
    return mean_rr

def evaluate_Hit_N(ranks):
    hit_1=[]
    hit_3=[]
    for rank_score in ranks:
        if rank_score<=1:
            hit_1.append(1)
        if rank_score<=3:
            hit_3.append(1)
    print("RL hit@1: ",sum(hit_1)/len(ranks))
    print("RL hit@3: ",sum(hit_3)/len(ranks))
if __name__=="__main__":

    featurePath = "./results_whole/"+dataset_name+"/"+task_name+"/"+'/cosine_total_path_stats_eval.txt'  # reasoning paths found the RL agent
    #mean_ap = evaluate_logic()
    mean_rr= evaluate_MRR()
