import os
from env_forbaseline import EnvDrones
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
import time
import pandas as pd


#### Change your number of nodes HERE! ####
num_nodes = 6
sr_rate = 100
#### Change your test times HERE! ####
test_times = 10
#### Change if you want to add noise later (default 0) ####
bdg_step = 0


def test(num_nodes, bdg_step, sr_rate):
    num_agents = num_nodes - 1
    SR_threshold = 100
    

    THR = []
    DEL = []
    LST = []


    # Test 10 times to calculate average amount
    for t in range(test_times):
        print('###############################################')
        print('Simulation start! sending rate is ', sr_rate)
        print('Time:', t+1)
        # Initialization
        env = EnvDrones(num_nodes, sr_rate)   
        # env.Init(t)
        env.Load_env(t)
        

        # Initialize evaluation matrix
        step = 0
        delays = [0.0] # sum of delays for all agents
        agent_delays  = [[0.0] for _ in range(num_agents)]  # individual agent delays
        # fig = plt.figure()
        # gs = GridSpec(2,2, figure=fig)
        # ax1 = fig.add_subplot(gs[0:1, 0:1])
        # ax2 = fig.add_subplot(gs[0:1, 1:2])
        # ax3 = fig.add_subplot(gs[1:2, 0:1])
        # ax4 = fig.add_subplot(gs[1:2, 1:2])
        # ax1.imshow(env.Get_full_obs_color())
        # ax2.imshow(env.Get_drone_obs_color(3))
        t_start = time.time()

        while True:
            # take step 
            delay_n = env.step()
            step += 1
            # ax3.imshow(env.Get_full_obs_color())
            # ax4.imshow(env.Get_drone_obs_color(3))
            # Collect delays for all agents
            for i, D in enumerate(delay_n):
                delays[-1] += D
                agent_delays[i][-1] += D

            # Check and break the simulation
            if step == SR_threshold or env.Check_finish():
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
                    break
                # If not, caculate THR percentage and enlarge DEL in the same proportion
                else:
                    print('Steps up! At {} seconds in real, and {} simulation steps.'.format(round(time.time() - t_start, 1), step))
                    print('Throughput is ', thr)
                    print('Throughput percentage is ',thr/(SR_threshold*sr_rate/100*env.max_sr))
                    THR.append([thr,thr/(SR_threshold*sr_rate/100*env.max_sr)]) 
                    print('Lost pkts are ', env.lost_pkt)
                    LST.append(env.lost_pkt)
                    print('Agent delays are ', agent_delays[:])
                    DEL.append(agent_delays)
                    print('Total Delay is ', delays[-1] / SR_threshold * (2400000/(sr_rate/100*env.max_sr)))
                    break
        
        # plt.draw()
        # plt.pause(10)
        # plt.close()

    writer = pd.ExcelWriter('./output/DEL_baseline_'+str(num_nodes)+str(sr_rate)+'.xlsx')
    pd.DataFrame(DEL).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()

    writer = pd.ExcelWriter('./output/THR_baseline_'+str(num_nodes)+str(sr_rate)+'.xlsx')
    pd.DataFrame(THR).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()

    writer = pd.ExcelWriter('./output/LST_baseline_'+str(num_nodes)+str(sr_rate)+'.xlsx')
    pd.DataFrame(LST).to_excel(writer, 'page_1', float_format = '%0.2f')
    writer.save()
    writer.close()


if __name__ == '__main__':
    test(num_nodes, bdg_step, sr_rate)





