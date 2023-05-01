
import os
import sys
import paddle
import argparse
# hyperparameters
state_dim = 200


parser=argparse.ArgumentParser(description='')
parser.add_argument("--gpu",type=str,default=None)
parser.add_argument("--max_step",type=int,default=200)
parser.add_argument("--dataset",type=int,default=0)
parser.add_argument("--relation",type=int,default=0)

parser.add_argument("--learningrate",type=float,default=0.001)
parser.add_argument("--alpha",type=float,default=0.05)
parser.add_argument("--beta",type=float,default=-0.005)

args=parser.parse_args()
add_info="lr_"+str(args.learningrate)+"_alpha_"+str(args.alpha)+"_beta"+str(args.beta)

if args.gpu:
    GPU_SET="gpu:"+str(args.gpu)
else:
    GPU_SET="cpu"
paddle.device.set_device(GPU_SET)

task_name_set_crd=['iscolor', 'hasshape', 'impactedby', 'hasstate', 'measureobject']
task_name_set_fb=['tvlanguage', 'birthplace', 'filmdirector', 'personnationality', 'capitalof', 'musicianorigin', 'organizationfounded', 'filmlanguage', 'filmwrittenby', 'teamsports']
task_name_set_nell=['concept_athletehomestadium', 'concept_athleteplaysinleague', 'concept_athleteplaysforteam','concept_teamplayssport', 'concept_personborninlocation', 'concept_organizationheadquarteredincity', 'concept_personleadsorganization', 'concept_organizationhiredperson', 'concept_worksfor', 'concept_athleteplayssport']
dataset=["FB15K-237","NELL-995","CRD"]
dataset_info={"FB15K-237":task_name_set_fb,"NELL-995":task_name_set_nell,"CRD":task_name_set_crd}
dataset_name=dataset[args.dataset]

tasks_set=dataset_info[dataset_name]

if dataset_name=="FB15K-237": 
    action_space = 474
    relation_real = {
        "birthplace":"/people/person/place_of_birth",
        "capitalof": "/location/country/capital",
        "filmdirector": "/film/director/film",
        "teamsports": "/sports/sports_team/sport",
        "personnationality": "/people/person/nationality",
        "tvlanguage": "/tv/tv_program/languages",
        "organizationfounded": "/organization/organization_founder/organizations_founded",
        "musicianorigin": "/music/artist/origin",
        "filmwrittenby": "/film/film/written_by",
        "filmlanguage": "/film/film/language",
    }
elif dataset_name=="NELL-995":
    action_space = 400
    relation_real={
    "concept_athletehomestadium":"concept:athletehomestadium",
    "concept_athleteplaysforteam":"concept:athleteplaysforteam",
    "concept_athleteplaysinleague":"concept:athleteplaysinleague",
    "concept_athleteplayssport":"concept:athleteplayssport",
    "concept_organizationheadquarteredincity":"concept:organizationheadquarteredincity",
    "concept_organizationhiredperson":"concept:organizationhiredperson",
    "concept_personborninlocation":"concept:personborninlocation",
    "concept_personleadsorganization":"concept:personleadsorganization",
    "concept_teamplayssport":"concept:teamplayssport",
    "concept_worksfor":"concept:worksfor",
    }
elif dataset_name=="CRD":
    action_space = 20
    relation_real={"hasshape":"hasShape",
                   "hasstate":"hasState",
                   "impactedby": "Impactedby",
                   "iscolor": "isColor",
                   "measureobject":"MeasureObject"}
task_name=tasks_set[args.relation]
relation=relation_real[task_name]

max_steps=args.max_step

from pyfiglet import Figlet
figlet=Figlet(font="standard")
print(figlet.renderText("Dataset:  "+dataset_name+"\nrelation: "+relation))

#read file path
DATASET="./Datasets/"+dataset_name+"/"
GRAPH_PATH= DATASET+ 'graph.txt'
KB_ENV_RL=DATASET+ 'kb_env_rl.txt'

entities_id=DATASET+"entity2id.txt"
relations_id=DATASET+"relation2id.txt"
entity_dim=DATASET+"entity_100dim"
relation_dim=DATASET+"relation_100dim"

#task file path
RELATION_TASK_TRAIN_POS = DATASET + 'tasks/' + task_name + '/' + 'train_pos'
RELATION_TASK_TRAIN_PAIRS = DATASET + 'tasks/' + task_name + '/' + 'train.pairs'
RELATION_TASK_TEST_PAIRS = DATASET + 'tasks/' + task_name + '/' + 'sort_test.pairs'
RELATION_TASK_TEST_POS = DATASET + 'tasks/' + task_name + '/' + 'test_pos'
#model save file
# PRE_TRAIN_PATH="./models/pre_model/"+dataset_name+"/"+task_name+"/"
# AGENT_TRAIN_PATH="./models/rl_model/"+dataset_name+"/"+task_name+"/"
PRE_TRAIN_PATH="./models/pre_model/"+dataset_name+"/"+task_name+"/"
AGENT_TRAIN_PATH="./models/rl_model/"+dataset_name+"/"+task_name+"/"

#train file path
if not os.path.exists(PRE_TRAIN_PATH):
    os.makedirs(PRE_TRAIN_PATH)
if not os.path.exists(AGENT_TRAIN_PATH):
    os.makedirs(AGENT_TRAIN_PATH)

#results file path
TASK_RESULT_PATH="./results/"+dataset_name+"/"+task_name+"/"
if not os.path.exists(TASK_RESULT_PATH):
    os.makedirs(TASK_RESULT_PATH)

POS_PATH = TASK_RESULT_PATH+add_info+"positive_paths.txt"
NEG_PATH = TASK_RESULT_PATH+add_info+"negative_paths.txt"
SHARE_PATH=TASK_RESULT_PATH+add_info+"sharing_paths.txt"

