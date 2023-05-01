
import paddle
import time
from tqdm import tqdm,trange

from Agent.di_agent import BaseAgent
from envs.env import Env

from config import *
from utils import *
paddle.device.set_device(GPU_SET)

if dataset_name=="FB15K-237":
    from data_analysis.entity_relation_fb import Analyse
elif dataset_name=="NELL-995":
    from data_analysis.entity_relation_crd import Analyse
elif dataset_name == "CRD":
    from data_analysis.entity_relation_nell import Analyse
def read_train_data():#
    train_head_entities=[]
    train_relations=[]
    train_tail_entities=[]
    with open(RELATION_TASK_TRAIN_POS,"r") as f:#[h,t,r]
        content=f.readlines()
        for line in content:
            temp=line.split()
            train_head_entities.append(temp[0])
            train_relations.append(temp[2])
            train_tail_entities.append(temp[1])
    return train_head_entities,train_relations,train_tail_entities

def pretrain():
    p_losss=[]
    n_losss=[]
    Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))

    pos_agent=BaseAgent("pos")
    neg_agent=BaseAgent("neg")

    f = open(RELATION_TASK_TRAIN_POS)
    train_data = f.readlines()
    f.close()
    num_samples = len(train_data)
    if num_samples > 500:
        num_samples = 500
        
    env = Env(relation)
    with trange(10) as pbar:
        for episode in pbar:
            print('Training Sample:', train_data[episode % num_samples][:-1])
            sample = train_data[episode % num_samples].split()
            good_episodes_pos = teacher(sample[0], sample[1], 5, env, GRAPH_PATH)
            good_episodes_neg = teacher(sample[1], sample[0], 5, env, GRAPH_PATH)
            if good_episodes_pos != []:
                for item in good_episodes_pos:
                    state_batch = []
                    action_batch = []
                    for t, transition in enumerate(item):
                        state_batch.append(transition.state)
                        action_batch.append(transition.action)
                        p_loss = pos_agent.agent.learn_state_action(transition.state, transition.action)
            else:
                if good_episodes_neg == []:
                    pass
                else:
                    new_good_episodes_pos = []
                    for episode in good_episodes_neg:
                        #print("good_episode", good_episodes_neg)
                        episode = episode[0]
                        good_episode = Transition(state=episode.next_state, action=episode.action, next_state=episode.state,
                                                reward=episode.reward, next_obs=episode.next_obs)
                        new_good_episodes_pos.append(good_episode)

                    for item in new_good_episodes_pos:
                        p_loss = pos_agent.agent.learn_state_action(transition.state, transition.action)


            if good_episodes_neg != []:
                for item in good_episodes_neg:
                    for t, transition in enumerate(item):
                        n_loss = neg_agent.agent.learn_state_action(transition.state, transition.action)

            else:
                if good_episodes_pos == []:
                    pass
                else:
                    new_good_episodes_neg = []
                    for episode in good_episodes_pos:
                        episode = episode[0]
                        good_episode = Transition(state=episode.next_state, action=episode.action, next_state=episode.state,
                                                reward=episode.reward, next_obs=episode.next_obs)
                        new_good_episodes_neg.append(good_episode)
                    #print("new_good_episodes_neg", len(new_good_episodes_neg), new_good_episodes_neg)
                    for item in new_good_episodes_neg:
                        n_loss = neg_agent.agent.learn_state_action(transition.state, transition.action)
            p_losss.append(p_loss)
            n_losss.append(n_loss)
            pbar.set_description("RL-pre-training:%i"%episode)
            pbar.set_postfix(p_loss=p_loss.cpu().numpy(),n_loss=n_loss.cpu().numpy())
    pos_agent.save_pre_model("pos"+add_info)
    neg_agent.save_pre_model("neg"+add_info)
    print('Model saved')



def path_reason():

    print("1.Data loading... ...")
    train_head_entities, train_relations, train_tail_entities = read_train_data()#
    num_samples = len(train_head_entities)
    num_episodes = num_samples 
    pos_agent=BaseAgent(name="pos"+add_info)
    neg_agent=BaseAgent(name="neg"+add_info)
    pos_agent.load_pre_model("pos")
    neg_agent.load_pre_model("neg")
    success_count=[]
    success=0
    ##
    env=Env(relation)
    f_suc=open(TASK_RESULT_PATH+add_info+"_success.txt","a")
    with trange(num_episodes) as pbar:
        for i in pbar:
            print("Eposide %d" % i, "is running... ...")
            entity_head = train_head_entities[i]
            entity_tail = train_tail_entities[i]
            existed_relation = train_relations[i]
            pos_done,pos_transition=pos_agent.run(env,entity_head,entity_tail)
            neg_done,neg_transition=neg_agent.run(env,entity_tail,entity_head)

            path_episodes_pos,path_episodes_neg=parameter_sharing(pos_transition,neg_transition)
            if path_episodes_pos and path_episodes_neg:
                print("Find same position ... ... \n")
                for path_episode in path_episodes_pos:
                    pos_agent.agent.learn_state_action(path_episode["states"],path_episode["actions"],reward=path_episode["rewards"][0])
                for path_episode in path_episodes_neg:
                    neg_agent.agent.learn_state_action(path_episode["states"],path_episode["actions"],reward=path_episode["rewards"][0])
            if neg_done or pos_done or path_episodes_neg:
                success+=1
            success_count.append(success)
            if i%50==0:
                pos_agent.save_model(name="pos",epoch=i)
                neg_agent.save_model(name="neg",epoch=i)
    
            f_suc.writelines([str(success),",","\t"])
    f_suc.writelines("\n")
    f_suc.close()
    pos_agent.save_model(name="pos",epoch=i)
    neg_agent.save_model(name="neg",epoch=i)



if __name__=="__main__":
    #print("Start pre-training---")
    pretrain()
    print("Start RL-train---")
    path_reason()
    print("Ending")