import os
import sys
import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from gym import spaces

# Copy multi_discrete from ~\MADDPG\maddpg\common
from multi_discrete import MultiDiscrete

''' drone_list is the drones encapsulation list: [Drone_order:[pos:[x,y], adjaency, coverage]] '''
class Drones(object):
    def __init__(self, nodes_position, updated_coverage, buffer_array, current_rate):
        self.pos = nodes_position
        self.coverage = updated_coverage 
        self.buffer = buffer_array
        self.rate = current_rate

class EnvDrones(object):
    def __init__(self, num, SR_rate):
        self.agents = num # the number of nodes
        self.map_size = 100 # pixels, bigger than 0 and int
        self.total_num_nodes = 40 # bigger than 0 and int
        self.nums_antenna = 4 # the number of beams
        self.init_antenna_faced_angle = 90 # in degree [0,180]
        self.init_antenna_half_coverage = 15 # in degree [15,45]
        self.init_awareness = 35 # decide initial beam distance
        self.init_radius = 4 # initial radius of side lobes
        self.max_half_coverage = 45 # maximum half beamwidth
        self.min_awareness_distance = 20 # minimum beam distance
        self.drone_list = [] # List contains all nodes
        self.agent_position = np.array([])
        self.bgd_position = np.array([])
        self.max_sr = 40000 # (50Mbps) (1250bit/pkt) Maximum sending rate [pkt/s] (To be the maximum x-axis range)
        self.init_rate = np.array([self.max_sr]*self.agents) # Bandwidth is the tx rate on each node, nodes can determine their own stratagies 
        self.buffer_array = np.zeros([self.agents,3]) # Initialize buffer [total, waiting, just_arrive]
        self.buffersize = 60000 # (75Mb) pkts in every hop nodes
        self.datasize = 2400000 # (3Gb) total pkts in transmission (can be fully tans under ideal scenario)
        self.sr_rate = SR_rate / 100 * self.max_sr # For results figure
        self.action_space = []
        self.lost_pkt = 0 # Summerize amount of lost packets

        self.observation_space = []
        for _ in range(self.agents-1):
            #### Change the default size of actions and observations if you modify the actions or observations later! ####
            self.action_space.append(MultiDiscrete([[0,4], [0,4], [0,4]])) # First [0,4] facing angle, Second [0,4] coverage, Third [0,4] rate
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(52,), dtype=np.float32))


####################

    def Init(self):
        # Generate agent's infos
        # [angle, halfbeamwidth, range] '''20.2-90`, 24.7-60`, 28.6-45`, 35-30`, R = sqrt(36750/angle)'''
        self.agent_coverage = np.array([[self.init_antenna_faced_angle, self.init_antenna_half_coverage, self.init_awareness]] * self.agents)
        self.agent_coverage[-1,:] = [0,0,0] 
        [self.agent_position, self.agent_coverage] = self.Agents_init(self.agents, self.agent_coverage, self.max_half_coverage)
        print('agent_coverage',self.agent_coverage)
        # Buffer of relay nodes
        for i in range(1, self.agents - 1):
            self.buffer_array[i,0] = self.buffersize
        # Buffer of source and Destination nodes
        self.buffer_array[0,0], self.buffer_array[0,1], self.buffer_array[-1,0] = self.datasize, self.datasize, self.datasize
        # Stop the destination node
        self.init_rate[-1] = 0
        for i in range(self.agents):
            if i == 0:
                temp_drone = Drones(self.agent_position[i], self.agent_coverage[i], self.buffer_array[i], self.sr_rate)
            else:
                temp_drone = Drones(self.agent_position[i], self.agent_coverage[i], self.buffer_array[i], self.init_rate[i])
            self.drone_list.append(temp_drone)
        
        # Generate backgrounds
        self.bgd_position = np.random.randint(self.map_size, size=(self.total_num_nodes, 2))
        # [x_position, y_position]
        self.bgd_position = np.resize(self.bgd_position,(self.total_num_nodes,2)).astype(int)
        # Change range from [0,100] to [0,99]
        self.bgd_position[np.where(self.bgd_position == 100)] = 99
        self.bgd_coverage = self.Antenna_update(self.bgd_position, np.array([]), np.array([]), self.max_half_coverage)
        
        # Generate grip array
        self.grid_array = self.Grid_init(self.agent_position, self.agent_coverage, self.bgd_position, self.bgd_coverage, self.map_size, self.init_radius)
        # np.set_printoptions(threshold=sys.maxsize)
        # print('***Initialization quick look***')
        # print('```Nodes_position```')
        # print('```[x, y]```')
        # print(self.agent_position[:,:])
        # print('```Antenna_coverage```')
        # print('```[angle, half_beamwidth, range]```')
        # print(self.agent_coverage[:,:])
        # print('```Max degree on grid map```')
        # print(np.amax(self.grid_array))
        # print(self.grid_array)

    def Save_env(self,t):
        with open('./saved_env/agt_'+str(t)+'.npy', 'wb') as f1:
            np.save(f1, self.agent_position, allow_pickle=True)
        with open('./saved_env/agtc_'+str(t)+'.npy', 'wb') as f2:
            np.save(f2, self.agent_coverage, allow_pickle=True)
        with open('./saved_env/bgd_'+str(t)+'.npy', 'wb') as f3:
            np.save(f3, self.bgd_position, allow_pickle=True)
        with open('./saved_env/bgdc_'+str(t)+'.npy', 'wb') as f4:
            np.save(f4, self.bgd_coverage, allow_pickle=True)
####################
    def Agents_init(self, nums, coverage, max_half_coverage):
        nodes_position = np.array([])
        pairs = np.array([])
        if nums == 5:
            x = [5, 23, 41, 59, 77]
        if nums == 7:
            x = [5, 18, 31, 44, 57, 70, 83]
        if nums == 10:
            x = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        for i in range(nums):
            nodes_position = np.append(nodes_position, [x[i], random.randint(35, 65)])
            if i != nums:
                pairs = np.append(pairs, [i, i+1])
        pairs = np.resize(pairs,(nums-1,2)).astype(int)
        nodes_position = np.resize(nodes_position,(nums,2)).astype(int)
        coverage_update = self.Antenna_update(nodes_position, pairs, coverage, max_half_coverage)
        return [nodes_position, coverage_update]

    def Antenna_update(self, nodes_position, pairs, antenna_coverage, max_half_coverage):
        # For agents
        if pairs.any():
            for i in range(len(pairs)):
                a, x1, y1 = pairs[i,0], nodes_position[pairs[i,0],0], nodes_position[pairs[i,0],1]
                b, x2, y2 = pairs[i,1], nodes_position[pairs[i,1],0], nodes_position[pairs[i,1],1]
                # calculate angles within 0-180
                # angle = math.atan2(y2-y1, x2-x1)/math.pi*180 - 90
                angle = math.atan2(y1-y2, x1-x2)/math.pi*180 - 90
                if angle < 0:
                    angle = 360 + angle
                antenna_coverage[i,0] = angle
        # For backgrounds
        else:
            antenna_coverage = np.array([[self.init_antenna_faced_angle, self.init_antenna_half_coverage, self.init_awareness]] * self.total_num_nodes)
            for i in range(len(nodes_position)):
                half_beamwidth = random.randint(15, 45)
                antenna_coverage[i,0], antenna_coverage[i,1], antenna_coverage[i,2]= random.randint(0, 360), half_beamwidth, self.Awareness_calc(half_beamwidth)
        return antenna_coverage

    def Awareness_calc(self, x):
        '''20.2-90`, 24.7-60`, 28.6-45`, 35-30`, R = sqrt(36750/angle)'''
        return math.sqrt(18375/x)

####################
# Only initialize agent's locations on grid
    def Grid_init_no_bgd(self, agent_position, agent_coverage, bgd_position, bgd_coverage, grid_size, circle_radius):
        grid_array = np.zeros([grid_size,grid_size])
        for i in range(len(agent_position[:,0])):
            x_position = agent_position[i, 0]
            y_position = agent_position[i, 1]
            # Color the grid map
            # Activated nodes have a circle interference area
            circle = self.Circle_positons(circle_radius, [x_position, y_position], grid_size)
            for [a,b] in circle:
                grid_array = self.Update_grid(grid_array, a, b, 1)
            # Activated pairs have an cone interference area
            cone = self.Cone_positons(x_position, y_position, agent_coverage[i, 0], agent_coverage[i, 1], agent_coverage[i, 2], grid_size)
            for [c,d] in cone:    
                if [c,d] in circle: continue
                grid_array = self.Update_grid(grid_array, c, d, 1)

        for i in range(len(bgd_position[:,0])):                
            grid_array[bgd_position[i,0],bgd_position[i,1]] = 1 # black
        for i in range(len(agent_position[:,0])):
            grid_array[agent_position[i,0],agent_position[i,1]] = 2 # blue
        return grid_array

# Initialize bgd's locations on grid
    def Add_bgd(self):
        for i in range(len(self.bgd_position[:,0])):
            x_position = self.bgd_position[i, 0]
            y_position = self.bgd_position[i, 1]
            # Color the grid map
            # Activated nodes have a circle interference area
            circle = self.Circle_positons(self.init_radius, [x_position, y_position], self.map_size)
            for [a,b] in circle:
                self.grid_array = self.Update_grid(self.grid_array, a, b, 1)
            # Activated pairs have an cone interference area
            cone = self.Cone_positons(x_position, y_position, self.bgd_coverage[i, 0], self.bgd_coverage[i, 1], self.bgd_coverage[i, 2], self.map_size)
            for [c,d] in cone:    
                if [c,d] in circle: continue
                self.grid_array = self.Update_grid(self.grid_array, c, d, 1)
        for i in range(len(self.bgd_position[:,0])):                
            self.grid_array[self.bgd_position[i,0],self.bgd_position[i,1]] = 1 # black
        for i in range(len(self.agent_position[:,0])):
            self.grid_array[self.agent_position[i,0],self.agent_position[i,1]] = 2 # blue


####################

    def Grid_init(self, agent_position, agent_coverage, bgd_position, bgd_coverage, grid_size, circle_radius):
        grid_array = np.zeros([grid_size,grid_size])
        for i in range(len(agent_position[:,0])):
            x_position = agent_position[i, 0]
            y_position = agent_position[i, 1]
            # Color the grid map
            # Activated nodes have a circle interference area
            circle = self.Circle_positons(circle_radius, [x_position, y_position], grid_size)
            for [a,b] in circle:
                grid_array = self.Update_grid(grid_array, a, b, 1)
            # Activated pairs have an cone interference area
            cone = self.Cone_positons(x_position, y_position, agent_coverage[i, 0], agent_coverage[i, 1], agent_coverage[i, 2], grid_size)
            for [c,d] in cone:    
                if [c,d] in circle:continue
                grid_array = self.Update_grid(grid_array, c, d, 1)
            
        for i in range(len(bgd_position[:,0])):
            x_position = bgd_position[i, 0]
            y_position = bgd_position[i, 1]
            # Color the grid map
            # Activated nodes have a circle interference area
            circle = self.Circle_positons(circle_radius, [x_position, y_position], grid_size)
            for [a,b] in circle:
                grid_array = self.Update_grid(grid_array, a, b, 1)
            # Activated pairs have an cone interference area
            cone = self.Cone_positons_bgd(x_position, y_position, bgd_coverage[i, 0], bgd_coverage[i, 1], bgd_coverage[i, 2], grid_size)
            for [c,d] in cone:    
                if [c,d] in circle:continue
                grid_array = self.Update_grid(grid_array, c, d, 1)
        for i in range(len(bgd_position[:,0])):                
            grid_array[bgd_position[i,0],bgd_position[i,1]] = 1 # black
        for i in range(len(agent_position[:,0])):
            grid_array[agent_position[i,0],agent_position[i,1]] = 2 # blue
        return grid_array

    def Circle_positons(self, radius, position, grid_size):
        ans = []
        x = position[0]
        y = position[1]
        # radius generation
        for a in range(x-radius,x+radius+1):
            if a < 0 or a >= grid_size: continue
            for b in range(y-radius,y+radius+1):
                if b < 0 or b >= grid_size: continue
                r = math.sqrt((x - a) ** 2 + (y - b) ** 2)
                if r <= radius:
                    ans.append([a,b])
        return ans

    # To find all main lobes for agent nodes
    def Cone_positons(self, x, y, angle, half_beamwidth, d, grid_size):
        ans = []
        min_angle, max_angle = angle - half_beamwidth , angle + half_beamwidth
        if angle < 90:
            if max_angle > 90:
                max_angle = 90
            if min_angle < 0:
                min_angle = 0
        else:
            if max_angle > 180:
                max_angle = 180
            if min_angle < 90:
                min_angle = 90
        min_rad, max_rad = math.radians(min_angle), math.radians(max_angle)
        if min_rad > math.pi:
            min_rad -= 2 * math.pi
        if max_rad > math.pi:
            max_rad -= 2 * math.pi    
        # radius generation  
        for a in range(x-d,x+d+1):
            if a < 0 or a >= grid_size: continue
            for b in range(y-d,y+d+1):
                if b < 0 or b >= grid_size: continue
                r = math.sqrt((x - a) ** 2 + (y - b) ** 2)         
                if r <= d:
                    if min_rad < max_rad and min_rad < math.atan2(a-x, b-y) < max_rad: 
                        ans.append([a,b])
                    if min_rad > max_rad and (-4 < math.atan2(a-x, b-y) < max_rad or min_rad < math.atan2(b-y, a-x) < 4): 
                        ans.append([a,b])
        return ans

    # To find all main lobes for bgd nodes, no need to consider directional antenna exceed boundary
    def Cone_positons_bgd(self, x, y, angle, half_beamwidth, d, grid_size):
        ans = []
        min_angle, max_angle = angle - half_beamwidth , angle + half_beamwidth
        min_rad, max_rad = math.radians(min_angle), math.radians(max_angle)
        if min_rad > math.pi:
            min_rad -= 2 * math.pi
        if max_rad > math.pi:
            max_rad -= 2 * math.pi    
        # radius generation  
        for a in range(x-d,x+d+1):
            if a < 0 or a >= grid_size: continue
            for b in range(y-d,y+d+1):
                if b < 0 or b >= grid_size: continue
                r = math.sqrt((x - a) ** 2 + (y - b) ** 2)         
                if r <= d:
                    if min_rad < max_rad and min_rad < math.atan2(a-x, b-y) < max_rad: 
                        ans.append([a,b])
                    if min_rad > max_rad and (-4 < math.atan2(a-x, b-y) < max_rad or min_rad < math.atan2(b-y, a-x) < 4): 
                        ans.append([a,b])
        return ans

    # Update grids if neccessary, operation == 1 means add degree, operation == 0 means minus degree
    def Update_grid(self, grids, a, b, operation):
        # Add Color
        if operation == 1:
            if grids[a][b] == 0: grids[a][b] = 3 # lightgreen
            if grids[a][b] == 1 or grids[a][b] == 2: return grids # skip nodes
            else: grids[a][b] += 1
        # Minus Color
        else:
            if grids[a][b] == 3: grids[a][b] = 0 # whitesmoke
            elif grids[a][b] == 0: return grids # return
            elif grids[a][b] == 1 or grids[a][b] == 2: return grids # skip nodes
            else: grids[a][b] -= 1
        return grids
        
####################
    def Color(self, num):
        # ['whitesmoke','black','blue','lightgreen','yellow','darkorange','red','darkred']
        if num >= 8:
            return [139,0,0] # darkred
        elif 8 > num >= 7:
            return [255,0,0] # red
        elif 7 > num >= 6:
            return [255,140,0] # orange
        elif 6 > num >= 5:
            return [255,255,0] # yellow
        elif 5 > num >= 3:
            return [202,255,112] # lightgreen
        elif num == 2:
            return [0,0,255] # blue
        elif num == 1:
            return [0,0,0] # black
        else:
            return [245,245,245] # whitesmoke

    # Same to 'Get_full_obs', just color the grids
    def Get_full_obs_color(self):
        obs = np.ones((self.map_size, self.map_size, 3))
        for row_number in range(self.map_size):
            for col_number in range(self.map_size):
                obs[row_number,col_number] = self.Color(self.grid_array[row_number][col_number])
        return obs.astype(int)

    # Same to 'Get_drone_obs', just color the grids
    def Get_drone_obs_color(self, i):
        obs = np.ones((self.map_size, self.map_size, 3))
        drone = self.drone_list[i]
        angle = drone.coverage[0]  #[-90,90]
        coverage = drone.coverage[1]
        radius = drone.coverage[2]
        # Find node surroundings
        for row_number in range(int(-self.init_radius/2), int(self.init_radius/2)):
            for col_number in range(int(-self.init_radius/2), int(self.init_radius/2)):
                if math.sqrt((row_number) ** 2 + (col_number) ** 2) > self.init_radius:
                    continue
                x = drone.pos[0] + row_number
                y = drone.pos[1] + col_number
                # Make sure within map
                if x < 0 or x > self.map_size - 1 or y < 0 or y > self.map_size - 1:
                    continue    
                obs[x,y] = self.Color(self.grid_array[x][y])
        # Find single antenna's obs
        for row_number in range(radius):
            for col_number in range(radius):
                if angle < 0:
                    x = drone.pos[0] + row_number
                    y = drone.pos[1] - col_number
                    atan2 = -math.atan2(-row_number, col_number) - math.pi/2 # This maybe correct!
                if angle >= 0:
                    x = drone.pos[0] + col_number
                    y = drone.pos[1] + row_number
                    atan2 = math.atan2(col_number, row_number) # This maybe correct!
                d = math.sqrt((row_number) ** 2 + (col_number) ** 2)
                # Reduce from rectangle to circle
                if d > radius or d < self.init_radius:
                    continue
                # Make sure within map
                if x < 0 or x > self.map_size - 1 or y < 0 or y > self.map_size - 1:
                    continue
                # Find the sector
                if math.radians(angle-coverage) <= atan2 <= math.radians(angle+coverage):
                    obs[x,y] = self.Color(self.grid_array[x][y])
                else:
                    continue
        return obs.astype(int)

####################
    
    # Return whole grids
    def Get_full_obs(self):
        return self.grid_array
    

    # Return 7*7 pixels observation surrounding receivers
    def Get_drone_obs(self, i):
        obs = np.zeros((7,7))
        if i < self.agents - 1:
            next_drone = self.drone_list[i+1]
            for a in range(7):
                for b in range(7):
                    obs[a,b] = self.grid_array[next_drone.pos[0]+a-3][next_drone.pos[1]+b-3]
            return obs
        return obs

    # Returns the observation of 7*7 pixels in the quadrant pointing to the transmitter direction
    def Get_drone_obs_toward(self, i):
        obs = np.zeros((7,7))
        if i < self.agents - 1:
            next_drone = self.drone_list[i+1]
            x_1, y_1 = self.drone_list[i].pos[0], self.drone_list[i].pos[1]
            x_2, y_2 = self.drone_list[i+1].pos[0], self.drone_list[i+1].pos[1]
            if x_1 < x_2:
                for a in range(7):
                    for b in range(7):
                        obs[a,b] = self.grid_array[next_drone.pos[0]-a][next_drone.pos[1]-b]
                return obs
            else:
                for a in range(7):
                    for b in range(7):
                        obs[a,b] = self.grid_array[next_drone.pos[0]+a][next_drone.pos[1]-b]
                return obs
        return obs

    # Return buffersize of receivers observations, rate here is only for calculating delays.
    def Get_buffer_obs(self, order):
        # Find downstream node
        if order < self.agents - 1:
            return np.concatenate((self.drone_list[order+1].buffer[:-1], self.drone_list[order].rate), axis = None)
        return np.concatenate((np.array([0,0]), self.drone_list[order].rate), axis = None)

    # Check if all data has been transmitted
    def Check_finish(self):
        ans = 0
        for i in range(self.agents-1):
            # If source node has no data and relay nodes have no buffer remains
            if int(self.drone_list[i].buffer[1]) != 0:
                ans += 1
        if ans != 0:
            return False
        return True

    # Update and calculate actions, observations, rewards, delays, grids and buffers
    def step(self, action_n):
        # action_n format: [[a*5,b*5,c*5]*agents] -- a for facing angle, b for coverage selection, c for rate
        actions = [[] for _ in range(self.agents-1)]
        # print('action_n',action_n)
        for a, action in enumerate(action_n):
            # print(action)
            max1, max2, max3 = [0,0],[0,5],[0,10]
            for b, act in enumerate(action):
                if 0 <= b <= 4:
                    if max1[0] < act:
                        max1 = [act,b]
                if 5 <= b <= 9:
                    if max2[0] < act:
                        max2 = [act,b]
                if 10 <= b <= 14:
                    if max3[0] < act:
                        max3 = [act,b]
            actions[a].append(max1[1])
            actions[a].append(max2[1])
            actions[a].append(max3[1])
        print(actions)    

        # actions format: [[x,y,z]*agents] - x is [0,4] y is [5,9] z is [10,14]
        for order, agent in enumerate(self.drone_list):
            # No actions on last/destination node
            if order == self.agents - 1:
                break
            # delta x and delta y for following checks
            delta_x, delta_y = self.drone_list[order+1].pos[0]-agent.pos[0], self.drone_list[order+1].pos[1]-agent.pos[1]
            # Change antenna coverage and range
            # Range of actions[i][1] is [5,6,7,8,9], 5 means doing nothing
            old_coverage, old_range = agent.coverage[1], agent.coverage[2]
            if actions[order][1] != 5:
                [agent.coverage[1], agent.coverage[2]] = self._select_antenna(actions[order][1]-5)
                # Forbiden no enough coverage range after actions
                if agent.coverage[2] < math.sqrt((delta_x) ** 2 + (delta_y) ** 2):
                    agent.coverage[1], agent.coverage[2] = old_coverage, old_range

            # Change face angle
            # Range of actions[i][0] is [0,1,2,3,4], 0 means doing nothing
            old_faceangle = agent.coverage[0]
            if actions[order][0] != 0:
                agent.coverage[0] = self._set_facing_angle(action_n[order][0], old_faceangle)
                # Forbiden no coverage area after actions
                center_angle = math.atan2(delta_x, delta_y)/math.pi*180 
                if agent.coverage[0] + agent.coverage[1] < center_angle or agent.coverage[0] - agent.coverage[1] > center_angle:
                    agent.coverage[0] = old_faceangle

            # Update grid map
            if old_faceangle != agent.coverage[0] or old_coverage != agent.coverage[1]:
                cone_old = self.Cone_positons(agent.pos[0], agent.pos[1], old_faceangle, old_coverage, old_range, self.map_size)
                cone = self.Cone_positons(agent.pos[0], agent.pos[1], agent.coverage[0], agent.coverage[1], agent.coverage[2], self.map_size)
                for [a,b] in cone_old:
                    self.grid_array = self.Update_grid(self.grid_array, a, b, 0)
                for [c,d] in cone:
                    self.grid_array = self.Update_grid(self.grid_array, c, d, 1)
        obs_n = []
        reward_n = []
        delay_n = []

        # Update buffer and rate (Tx)
        for i, agent in enumerate(self.drone_list):
            if i < self.agents - 1 and agent.buffer[1] != 0:
                # agent.rate = self.max_sr * max((-0.13*Check_surrounding(self, agent.pos[0], agent.pos[1]) + 1.39), 0.1)
                # Rate changes following by actions 25%, 50%, 75%, 100%
                # Range of actions[i][2] is [10,11,12,13,14], 10 means doing nothing
                # SR of source node is firm (self.sr_rate)
                if i == 0:
                    agent.rate = int(self.sr_rate)
                else:
                    if actions[i][2] != 10:
                        agent.rate = int(self.max_sr * (actions[i][2]-10) * 0.25)
                # Make sure that buffer won't exceed and won't be negative
                if agent.rate > agent.buffer[1]:
                    agent.rate = int(agent.buffer[1])
                # Trans data
                agent.buffer[1] -= int(agent.rate)
                # self.drone_list[i+1].buffer[2] += int(agent.rate * (1 - max(min(((self.Check_surrounding(self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0)))
                self.drone_list[i+1].buffer[2] += int(agent.rate * (1 - max(min(((self.Check_surrounding_toward(self.drone_list[i].pos[0],self.drone_list[i].pos[1], self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0)))
                # self.lost_pkt += int(agent.rate * max(min(((self.Check_surrounding(self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0))
                self.lost_pkt += int(agent.rate * max(min(((self.Check_surrounding_toward(self.drone_list[i].pos[0],self.drone_list[i].pos[1], self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0))
                # Update buffer for calculating delays
                if self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] > self.drone_list[i+1].buffer[0]:
                    self.lost_pkt += int(agent.rate + self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] - self.drone_list[i+1].buffer[0])
                    self.drone_list[i+1].buffer[2] = int(self.drone_list[i+1].buffer[0] - self.drone_list[i+1].buffer[1])

        # Get obs, reward and delay based on new infos (Rx)
        for j, agent in enumerate(self.drone_list):
            if j < self.agents - 1:
                obs_n.append(self._get_obs(j))
                delay_n.append(self._get_delay(j, agent))
                # Update buffer again
                agent.buffer[1] += agent.buffer[2]
                agent.buffer[2] = 0
                reward_n.append(self._get_reward(j, agent))

        return obs_n, reward_n, delay_n


    def step_baseline(self):
        delay_n = []
        # Update buffer and rate (Tx)
        for i, agent in enumerate(self.drone_list):
            if i < self.agents - 1 and agent.buffer[1] != 0:

                # Make sure that buffer won't exceed and won't be negative
                if agent.rate > agent.buffer[1]:
                    agent.rate = agent.buffer[1]

                # Check interferences and trans data
                agent.buffer[1] -= int(agent.rate)
                self.drone_list[i+1].buffer[2] += int(agent.rate * (1 - max(min(((self.Check_surrounding(self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0)))
                self.lost_pkt += int(agent.rate * max(min(((self.Check_surrounding(self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1]) - 4) / 6), 1),0))
                # Update buffer for calculating delays
                if self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] > self.drone_list[i+1].buffer[0]:
                    self.lost_pkt += int(agent.rate + self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] - self.drone_list[i+1].buffer[0])
                    self.drone_list[i+1].buffer[2] = int(self.drone_list[i+1].buffer[0] - self.drone_list[i+1].buffer[1])

        # Get obs, reward and delay based on new infos (Rx)
        for j, agent in enumerate(self.drone_list):
            if j < self.agents - 1 and j != 0:
                delay_n.append(self._get_delay(j, agent))
                # Update buffer again
                agent.buffer[1] += agent.buffer[2]
                agent.buffer[2] = 0

        return delay_n




    # Facing angle: [24,156], based on the possible positions
    def _set_facing_angle(self, num, old_angle):
        if old_angle >= 90:
            if num == 1:
                return max(90, old_angle - 10)
            elif num == 2:
                return max(90, old_angle - 5)
            elif num == 3:
                return min(156, old_angle + 5)
            else:
                return min(156, old_angle + 10)
        else:
            if num == 1:
                return max(24, old_angle - 10)
            elif num == 2:
                return max(24, old_angle - 5)
            elif num == 3:
                return min(89, old_angle + 5)
            else:
                return min(89, old_angle + 10)

    def _select_antenna(self, num):
        ### 20.2-90`, 24.7-60`, 28.6-45`, 35-30`, R = sqrt(36750/angle) ###
        if num == 1:
            return [45, 20.2]
        elif num == 2:
            return [30, 24.7]
        elif num == 3:
            return [22.5, 28.6]
        else:
            return [15, 35]

    def _get_obs(self, i):
        # return np.concatenate((self.Get_drone_obs(i), self.Get_buffer_obs(i)), axis = None)
        
        return np.concatenate((self.Get_drone_obs_toward(i), self.Get_buffer_obs(i)), axis = None)
    # Return agent's rewards
    def _get_reward(self, i, agent):
        if i < self.agents - 1:
            ans = 0
            # Add 1 if average grid degree is below 4 (best), add 0 if grid degree is greater than 10 (worst)
            x, y = self.drone_list[i+1].pos[0], self.drone_list[i+1].pos[1]
            # ans += 1 - max(min(((self.Check_surrounding(x,y) - 4) / 6), 1),0)

            # To change surrounding to toward
            a, b = agent.pos[0], agent.pos[1]
            ans += 0 - max(min(((self.Check_surrounding_toward(a, b, x, y) - 4) / 6), 1),0)

            # Minus 1 if all buffer occupied (worst), minus 0 if buffer is empty (best)
            if i < self.agents - 2:
                ans -= self.drone_list[i+1].buffer[1]/self.drone_list[i+1].buffer[0]
            # Encourage agents using highest trasmission rate
            if i < self.agents - 1:
                ans += agent.rate/self.max_sr
            return ans
        return 0

    # To calculate degrees of 7*7 pixels surrounding receiver for rewards
    def Check_surrounding(self, x, y):
        # Get 7*7 pixels surrounding nodes
        num = 0
        for a in range(x-3, x+4):
            for b in range(y-3, y+4):
                num += self.grid_array[a][b]
        # Minus degree of [x][y] (which is 2), divided by (49-1)
        return (num - 2)/48

    # Returns the observation of 7*7 pixels in the quadrant pointing to the transmitter direction for rewards
    def Check_surrounding_toward(self, x_1, y_1, x_2, y_2):
        # Get 7*7 pixels toward transmitter
        num = 0
        if x_1 < x_2:
            for a in range(x_2-7, x_2):
                for b in range(y_2-6, y_2+1):
                    num += self.grid_array[a][b]
        else:
            for a in range(x_2, x_2+7):
                for b in range(y_2-6, y_2+1):
                    num += self.grid_array[a][b]           
        # Minus degree of [x][y] (which is 2), divided by (49-1)
        return (num - 2)/48


    # Return delays for agents
    def _get_delay(self, i, agent):
        if i != 0 and agent.rate != 0:
            return agent.buffer[1]/agent.rate
        return 0
