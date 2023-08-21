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





''' drone_list is the drones encapsulation list: [Drone_order:[pos:[x,y], adjaency, coverage]] '''
class Drones(object):
    def __init__(self, nodes_position, updated_coverage, buffer_array, current_rate):
        self.pos = nodes_position
        self.coverage = updated_coverage # Facing angle: [24,156]
        self.buffer = buffer_array
        self.rate = current_rate

class EnvDrones(object):
    def __init__(self, num, SR_rate):
        self.agents = num # the number of nodes
        self.map_size = 100 # pixels, bigger than 0 and int
        self.total_num_nodes = 40 # bigger than 0 and int
        self.nums_antenna = 4
        self.init_antenna_faced_angle = 0 # in degree [-75,75]
        self.init_antenna_half_coverage = 15 # in degree [15,45]
        self.init_awareness = 35
        self.init_radius = 4
        self.max_half_coverage = 45
        # self.min_awareness_distance = 20
        # self.edges_probability = 0.08 # 0-1
        self.drone_list = []
        self.agent_position = np.array([])
        self.bgd_position = np.array([])
        self.time_resolution = 1 # Time resolution[s]
        self.max_sr = 40000 # (50Mbps)  (1250bit/pkt) Maximum sending rate [pkt/s] (To be the maximum x-axis range)
        self.init_rate = np.array([self.max_sr]*self.agents) # Bandwidth is the tx rate on each node, nodes can determine their own stratagies 
        self.buffer_array = np.zeros([self.agents,3]) # Initialize buffer [total, waiting, just_arrive]
        # self.buffersize = 60000 # (75Mb) pkts in every hop nodes
        self.buffersize = 120000 # (150Mb) pkts in every hop nodes
        self.datasize = 2400000 # (3Gb) total pkts in transmission (can be fully tans under ideal scenario)
        self.sr_rate = SR_rate / 100 * self.max_sr # For results figure
        self.action_space = []
        self.lost_pkt = 0

        self.observation_space = []


####################

    # def Init(self,t):
    #     # Generate agent's infos
    #     # [angle, halfbeamwidth, range] '''20.2-90`, 24.7-60`, 28.6-45`, 35-30`, R = sqrt(36750/angle)'''
    #     self.agent_coverage = np.array([[self.init_antenna_faced_angle, self.init_antenna_half_coverage, self.init_awareness]] * self.agents)
    #     self.agent_coverage[-1,:] = [0,0,0] 
    #     [self.agent_position, self.agent_coverage] = self.Agents_init(self.agents, self.agent_coverage, self.max_half_coverage)
        

    #     # Buffer of relay nodes
    #     for i in range(1, self.agents - 1):
    #         self.buffer_array[i,0] = self.buffersize
    #     # Buffer of source and Destination nodes
    #     self.buffer_array[0,0], self.buffer_array[0,1], self.buffer_array[-1,0] = self.datasize, self.datasize, self.datasize
    #     # Stop the destination node
    #     self.init_rate[-1] = 0
    #     for i in range(self.agents):
    #         if i == 0:
    #             temp_drone = Drones(self.agent_position[i], self.agent_coverage[i], self.buffer_array[i], self.sr_rate)
    #         else:
    #             temp_drone = Drones(self.agent_position[i], self.agent_coverage[i], self.buffer_array[i], self.init_rate[i])
    #         self.drone_list.append(temp_drone)
    #     # Generate backgrounds
    #     self.bgd_position = np.random.randint(self.map_size, size=(self.total_num_nodes, 2))
    #     # [x_position, y_position]
    #     self.bgd_position = np.resize(self.bgd_position,(self.total_num_nodes,2)).astype(int)
    #     # Change range from [0,100] to [0,99]
    #     self.bgd_position[np.where(self.bgd_position == 100)] = 99
    #     self.bgd_coverage = self.Antenna_update(self.bgd_position, np.array([]), np.array([]), self.max_half_coverage)
        
    #     # Generate grid array
    #     # Load grid for baseline testing
    #     # with open(str(t)+'.npy', 'rb') as f:
    #     #     self.grid_array = np.load(f, allow_pickle=True)
    #     self.grid_array = self.Grid_init(self.agent_position, self.agent_coverage, self.bgd_position, self.bgd_coverage, self.map_size, self.init_radius)

    #     # np.set_printoptions(threshold=sys.maxsize)
    #     # print('***Initialization quick look***')
    #     # print('```Nodes_position```')
    #     # print('```[x, y]```')
    #     # print(self.agent_position[:,:])
    #     # print('```Antenna_coverage```')
    #     # print('```[angle, half_beamwidth, range]```')
    #     # print(self.agent_coverage[:,:])
    #     # print('```Max degree on grid map```')
    #     # print(np.amax(self.grid_array))
    #     # print(self.grid_array)
    #     # print(self.buffer_array)

    def Load_env(self,t):
        with open('./saved_env/agt_'+str(t)+'.npy', 'rb') as f1:
            self.agent_position = np.load(f1, allow_pickle=True)
        with open('./saved_env/agtc_'+str(t)+'.npy', 'rb') as f2:
            self.agent_coverage = np.load(f2, allow_pickle=True)
        with open('./saved_env/bgd_'+str(t)+'.npy', 'rb') as f3:
            self.bgd_position = np.load(f3, allow_pickle=True)
        with open('./saved_env/bgdc_'+str(t)+'.npy', 'rb') as f4:
            self.bgd_coverage = np.load(f4, allow_pickle=True)

                 
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
        self.grid_array = self.Grid_init(self.agent_position, self.agent_coverage, self.bgd_position, self.bgd_coverage, self.map_size, self.init_radius)
####################
    def Agents_init(self, nums, coverage, max_half_coverage):
        nodes_position = np.array([])
        pairs = np.array([])
        if nums == 4:
            x = [5, 23, 41, 59]
        if nums == 6:
            x = [5, 18, 31, 44, 57, 70]
        if nums == 8:
            x = [5, 15, 25, 35, 45, 55, 65, 75]
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
            cone = self.Cone_positons(x_position, y_position, bgd_coverage[i, 0], bgd_coverage[i, 1], bgd_coverage[i, 2], grid_size)
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

    def Cone_positons(self, x, y, angle, half_beamwidth, d, grid_size):
        ans = []
        min_angle, max_angle = angle - half_beamwidth - 90, angle + half_beamwidth - 90
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
                    if min_rad < max_rad and min_rad < math.atan2(b-y, a-x) < max_rad: 
                        ans.append([a,b])
                    if min_rad > max_rad and (-4 < math.atan2(b-y, a-x) < max_rad or min_rad < math.atan2(b-y, a-x) < 4): 
                        ans.append([a,b])
        return ans

    def Update_grid(self, grids, a, b, operation):
        # Add Color
        if operation == 1:
            if grids[a][b] == 0: grids[a][b] = 3 # lightgreen
            else: grids[a][b] += 1
        # Minus Color
        else:
            if grids[a][b] == 3: grids[a][b] = 0 # whitesmoke
            elif grids[a][b] == 0: return grids # return
            else: grids[a][b] -= 1
        return grids

####################
# This part is for visualization

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

    def Get_full_obs_color(self):
        obs = np.ones((self.map_size, self.map_size, 3))
        for row_number in range(self.map_size):
            for col_number in range(self.map_size):
                obs[row_number,col_number] = self.Color(self.grid_array[row_number][col_number])
        return obs.astype(int)

    def Get_drone_obs_color(self, i):
        obs = np.ones((self.map_size, self.map_size, 3))
        drone = self.drone_list[i]
        angle = drone.coverage[0] - 90 #[-90,90]
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
                    atan2 = -math.atan2(-row_number, col_number) - math.pi/2 # This is correct!
                if angle >= 0:
                    x = drone.pos[0] + col_number
                    y = drone.pos[1] + row_number
                    atan2 = math.atan2(row_number, col_number) # This is correct!
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

    def Check_finish(self):
        ans = 0
        for i in range(self.agents-1):
            # If source node has no data and relay nodes have no buffer remains
            if int(self.drone_list[i].buffer[1]) != 0:
                ans += 1
        if ans != 0:
            return False
        return True
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


    def step(self):
        delay_n = []
        # Update buffer and rate (Tx)
        for i, agent in enumerate(self.drone_list):
            if i < self.agents - 1 and agent.buffer[1] != 0:

                # Make sure that buffer won't exceed and won't be negative
                if agent.rate > agent.buffer[1]:
                    agent.rate = int(agent.buffer[1])

                # Check interferences and trans data
                agent.buffer[1] -= int(agent.rate)


                # from 4 to 10 or more
                interference_degree = self.Check_surrounding_toward(self.drone_list[i].pos[0],self.drone_list[i].pos[1], self.drone_list[i+1].pos[0],self.drone_list[i+1].pos[1])
                # from 0 to 1
                percentage = (interference_degree - 4) / 6



                self.drone_list[i+1].buffer[2] += int(agent.rate * (1 - max(min(percentage, 1),0)))
                self.lost_pkt += int(agent.rate * max(min(percentage, 1),0))
                # Update buffer for calculating delays
                if self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] > self.drone_list[i+1].buffer[0]:
                    self.lost_pkt += int(self.drone_list[i+1].buffer[1] + self.drone_list[i+1].buffer[2] - self.drone_list[i+1].buffer[0])
                    self.drone_list[i+1].buffer[2] = int(self.drone_list[i+1].buffer[0] - self.drone_list[i+1].buffer[1])




        # Get obs, reward and delay based on new infos (Rx)
        for j, agent in enumerate(self.drone_list):
            if j < self.agents - 1 and j != 0:
                delay_n.append(self._get_delay(j, agent))
                # Update buffer again
                agent.buffer[1] += agent.buffer[2]
                agent.buffer[2] = 0

        return delay_n


    def Check_surrounding(self, x, y):
        # Get 7*7 pixels surrounding nodes
        num = 0
        for a in range(x-3, x+4):
            for b in range(y-3, y+4):
                num += self.grid_array[a][b]
        # Minus degree of [x][y] (which is 2), divided by (49-1)
        return (num - 2)/48

    def _get_delay(self, i, agent):
        if i != 0 and agent.rate != 0:
            return agent.buffer[1]/agent.rate
        return 0
