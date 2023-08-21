import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.contrib.layers as layers
# from tensorflow.compat.v1 import layers
# import tensorflow.contrib.layers as layers
from tf_slim import layers

from env_fortest import EnvDrones
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
import time
import argparse
import pandas as pd
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer


#### Change your number of nodes HERE! ####
num_nodes = 6
sr_rate = 100

#### Change your test times HERE! ####
test_times = 10


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--max_episode_len", type=int, default=5, help="maximum episode length")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer")
    parser.add_argument("--lr2", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--num_units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--SR_threshold", type=int, default=100, help="maximum steps length")
    parser.add_argument("--save-dir", type=str, default="./tmp/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--restore", action="store_true", default=True)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=16, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        # out = layers.Dense(out, num_units, activation=tf.nn.relu)
        # out = layers.Dense(out, num_units, activation=tf.nn.relu)
        # out = layers.Dense(out, num_outputs, activation=None)
        return out

def get_trainers(env, num_agents, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_agents):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='maddpg')))
    return trainers

def test(arglist, num_nodes, sr_rate):
    num_agents = num_nodes - 1

    THR = []
    REW = []
    DEL = []
    LST = []

    ARRAY = np.array([])
    # Test 10 times to calculate average amount
    for t in range(test_times):
        tf.reset_default_graph()
        print('###############################################')
        print('Simulation start! sending rate is ', sr_rate)
        print('Times:', t+1)
        with U.single_threaded_session():
            #Initialize evaluation matrix
            step = 0
            delays = [0.0] # sum of delays for all agents
            agent_delays  = [[0.0] for _ in range(num_agents)]  # individual agent delays
            rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(num_agents)]  # individual agent reward
            
            # Initialization env
            env = EnvDrones(num_nodes, sr_rate)   
            env.Init()
            env.Save_env(t)            
            # Get intial obs
            obs_n = []
            for i in range(num_agents):
                obs_n.append(env._get_obs(i))            
            
            t_start = time.time()
            obs_shape_n = [env.observation_space[i].shape for i in range(num_agents)]
            trainers = get_trainers(env, num_agents, obs_shape_n, arglist)
            
            U.initialize()
            # Load previous results, if necessary
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.restore:
                print('Loading previous state...')
                U.load_state(arglist.load_dir)

            # Loop until steps up or all data is transferred
            while True:
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                new_obs_n, rew_n, delay_n = env.step(action_n)
                step += 1

                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i])
                obs_n = new_obs_n
                
                # Collect delays for all agents
                for i, D in enumerate(delay_n):
                    delays[-1] += D
                    agent_delays[i][-1] += D

                # Collect rewards for all agents
                for i, rew in enumerate(rew_n):
                    rewards[-1] += rew
                    agent_rewards[i][-1] += rew
                rewards.append(0)

                # update all trainers, if not in display or benchmark mode
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, step)

                # Check and break the simulation
                if step == arglist.SR_threshold or env.Check_finish():
                    print(env.buffer_array)
                    thr = env.drone_list[-1].buffer[2]
                    # If done within time threshold
                    if env.Check_finish():
                        print('At {} seconds in real, and {} simulation steps.'.format(round(time.time() - t_start, 1), step))
                        print('Throughput is ', thr)
                        print('Throughput percentage is ',thr/2400000)
                        THR.append([thr,thr/2400000]) 
                        print('Lost pkts are ', env.lost_pkt)
                        LST.append(env.lost_pkt)
                        print('Agent delays are ', agent_delays[:])
                        DEL.append(agent_delays)
                        print('Total Delay is ', delays[-1])
                        print('Agent rewards are ', agent_rewards[:])
                        REW.append(agent_rewards)
                        # print('Total reward is ', rewards[-2])
                        # print('Whole reward is ', rewards)
                        break
                    # If not, caculate THR percentage and enlarge DEL in the same proportion
                    else:
                        print('Steps up! At {} seconds in real, and {} simulation steps.'.format(round(time.time() - t_start, 1), step))
                        print('Throughput is ', thr)
                        print('Throughput percentage is ',thr/(arglist.SR_threshold*sr_rate/100*env.max_sr))
                        THR.append([thr,thr/(arglist.SR_threshold*sr_rate/100*env.max_sr)]) 
                        print('Lost pkts are ', env.lost_pkt)
                        LST.append(env.lost_pkt)
                        print('Agent delays are ', agent_delays[:])
                        DEL.append(agent_delays)
                        print('Total Delay is ', delays[-1] / arglist.SR_threshold * (2400000/(sr_rate/100*env.max_sr)))
                        print('Agent rewards are ', agent_rewards[:])
                        REW.append(agent_rewards)
                        # print('Total reward is ', rewards[-2])
                        # print('Whole reward is ', rewards)
                        break

    writer = pd.ExcelWriter('./output/DEL_test_'+str(num_nodes)+'_'+str(sr_rate)+'.xlsx')
    pd.DataFrame(DEL).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()

    writer = pd.ExcelWriter('./output/THR_test_'+str(num_nodes)+'_'+str(sr_rate)+'.xlsx')
    pd.DataFrame(THR).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()

    writer = pd.ExcelWriter('./output/LST_test_'+str(num_nodes)+'_'+str(sr_rate)+'.xlsx')
    pd.DataFrame(LST).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()

    writer = pd.ExcelWriter('./output/REW_test_'+str(num_nodes)+'_'+str(sr_rate)+'.xlsx')
    pd.DataFrame(REW).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()


if __name__ == '__main__':
    arglist = parse_args()
    test(arglist, num_nodes, sr_rate)
