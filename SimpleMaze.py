# import clingo
import numpy as np

import copy
import random
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch import optim
import math
from HierarchicalAgentFlatQ import *
import pandas as pd
class Recursive_Maze():
    def __init__(self, shape_maze,simple,manager_layers,man_view,search_clause=False):
        self.reward_structure=simple
        self.n_layers = manager_layers
        self.manager_view=man_view
        self.search_clause=search_clause
        self.dim_maze = shape_maze
        self.generate_maze(self.dim_maze, simple)
        #         self.goal_state()
        self.initiate_goal_state()
        self.initiate_states()
        self.loc = self.agent_init_state
        self.ns = self.maze.shape[0] ** 2
        self.na = 4
        self.init_maze_hierarchy()
        self.current_level = 0
        self.current_tasks_loc = copy.copy(self.super_managers)
        self.tasks=[4 for x in self.current_tasks_loc]
        self.hierarchy_actions = [4 for x in range(int(self.n_layers))]
        #         need to keep track if we are in the right location according to our super manager
        self.tasks_bools = np.ones(len(self.current_tasks_loc))
        self.lims = self.get_super_manager_1([self.maze.shape[1], self.maze.shape[1]])
        self.search_lims = [4*x[0] for x in self.lims][::-1]
        # self.search_lims = [12000 for x in self.lims][::-1]
        # self.state_visit=np.zer
        self.search_lims[-1]=np.maximum(4,self.search_lims[-1])
        self.current_state=self.super_managers[self.current_level]
        # print(self.search_lims)
        self.reward_per_level = [0 for x in range(int(self.n_layers+1))]
        self.reset_reward= [0 for x in range(int(self.n_layers+1))]
        self.check_dicts={}
        self.expected_level=0
        self.check_dicts[0] = {}
        # self.check_dicts[0]=[0,0]
        for i in range(int(self.n_layers+1)):
            self.check_dicts[0][i]={0:0,1:0,2:0,3:0,4:0,5:0}
        # print(self.maze)
        # print(self.loc)
        assert self.maze[int(self.loc[0]), int(self.loc[1])] == 0
        assert self.maze[int(self.agent_init_state[0]), int(self.agent_init_state[1])] == 0

    def get_super_manager(self):

        super_managers = []
        number_of_levels = self.n_layers
        super_managers.append([np.floor(x / self.manager_view) for x in self.loc])
        if number_of_levels - 1 > 1:
            for i in range(int(number_of_levels - 1)):
                super_managers.append([np.floor(x / self.manager_view) for x in super_managers[-1]])
        #                 print(i)
        else:
            super_managers.append([0, 0])
        return super_managers[::-1]

    def initiate_goal_state(self):
        empty_cells = [[x, y] for x, y in zip(np.where(self.maze == 0)[0], np.where(self.maze == 0)[1])]
        self.goal_init_state=random.choice(empty_cells)

    def init_super_manager(self):
        n_layers = self.n_layers + 1
        self.super_managers = []
        number_of_levels = n_layers
        self.super_managers.append([np.floor(x / self.manager_view) for x in self.loc])
        if number_of_levels - 2 > 1:
            for i in range(int(number_of_levels - 2)):
                self.super_managers.append([np.floor(x / self.manager_view) for x in self.super_managers[-1]])
        else:
            self.super_managers.append([0, 0])
        self.super_managers = self.super_managers[::-1]

    def init_maze_hierarchy(self):
        # nlayers = math.log(self.maze.shape[0], 2)
        self.init_super_manager()
        number_of_levels=self.n_layers + 1
        self.goal_locs = []
        self.goal_locs.append([np.floor(x / self.manager_view) for x in self.goal_init_state])
        if number_of_levels - 2 > 1:
            for i in range(int(number_of_levels - 2)):
                self.goal_locs.append([np.floor(x / self.manager_view) for x in self.goal_locs[-1]])
        else:
            self.goal_locs.append([0, 0])
        self.goal_locs = self.goal_locs[::-1]

    def initiate_states(self):
        empty_cells = [[x, y] for x, y in zip(np.where(self.maze == 0)[0], np.where(self.maze == 0)[1])]
        self.agent_init_state = random.choice(empty_cells)

        if self.agent_init_state==self.goal_init_state:
            print('Reinitializing state')
            self.initiate_states()
        #         print(agent_init_state)

    def reset_rewards_after_learning(self,old_reset):
        for i,x in enumerate(old_reset):
            if x !=0:
                self.reward_per_level[i] = 0
                # if i!=self.n_layers:
                #     self.tasks_bools[i]=1

    # def check_if_manager_change(self, current_level,new_state,old_state,d):

    def check_if_task_satisfied_at_senior_level(self, current_level,new_state,old_state,d):
        locs = self.get_super_manager_1(self.loc)
        locs1 = self.get_super_manager_1(old_state)
        locs2 = self.get_super_manager_1(new_state)
        # print('Expected level 0', self.expected_level)
        if current_level != 0:
            if self.current_tasks_loc[current_level - 1] == locs[current_level - 1]:
                if locs1[current_level-1]!=locs2[current_level-1]:
                    if self.tasks_bools[current_level - 1] != 1:
                        if self.hierarchy_actions[current_level-1]==4:
                            if d:
                                print('task satisfied')

                                self.reset_reward[current_level - 1] = 5
                                self.reset_reward[current_level] = 5

                            else:
                                self.reset_reward[current_level - 1] = 0
                                self.reset_reward[current_level] = 0

                        else:
                            print('stask satisfied')
                            # print('level',current_level)
                            # print('moving from', old_state)
                            # print('movign to ', new_state)
                            # print('task', self.hierarchy_actions[current_level - 1])
                            # print('current taks  ', self.current_tasks_loc[current_level-1])
                            # print('task satisfied at level',current_level-1)
                            self.reset_reward[current_level-1]=1
                            self.reset_reward[current_level]=1

                        self.tasks_bools[current_level - 1] = 1
                        self.expected_level = np.maximum(self.expected_level - 1,0)
                        # print('Current level',current_level)
                        current_level = current_level - 1
                        # print('Expected level',self.expected_level)

                        self.check_if_task_satisfied_at_senior_level( current_level,new_state,old_state,d)
                    else:
                        self.expected_level=current_level
                        # print('Expected level TB', self.expected_level)
                else:
                    self.expected_level = current_level


            elif locs1 != locs2:
                if self.hierarchy_actions[current_level - 1] != 4:
                        print('moving from', old_state)
                        print('movign to ', new_state)
                        print('task',self.hierarchy_actions[current_level-1])
                        print('change managerial level', current_level - 1)

                        # if d:
                        #     self.reset_reward[current_level - 1] = 5
                        #     self.reset_reward[current_level] = 5
                        # else:
                        # print('TaskFailed')
                        self.reset_reward[current_level - 1] = 3
                        self.reset_reward[current_level] = 3
                        self.expected_level = np.maximum(self.expected_level - 1, 0)
                        # print('EL', self.expected_level)
                        current_level = current_level - 1
                        # self.tasks_bools[current_level - 1]=1
                        self.check_if_task_satisfied_at_senior_level(current_level, new_state, old_state, d)
                else:

                    print('moving from', old_state)
                    print('movign to ', new_state)
                    print('task', self.hierarchy_actions[current_level - 1])
                    print('change managerial level', current_level - 1)
                    # if d:
                    #     self.reset_reward[current_level - 1] = 5
                    #     self.reset_reward[current_level] = 5
                    # else:
                    # print('TaskFailed')
                    self.reset_reward[current_level - 1] = 3
                    self.reset_reward[current_level] = 3
                    self.expected_level = np.maximum(self.expected_level - 1, 0)
                    # print('EL',self.expected_level)
                    current_level = current_level - 1
                    # self.tasks_bools[current_level - 1] = 1

                    self.check_if_task_satisfied_at_senior_level(current_level, new_state, old_state, d)
            else:
                # print('no change')
                self.expected_level=current_level
                # print('EL', self.expected_level)
    def check_if_search_limit_breached(self, current_level,d):
        if current_level - 1 >= 0:
            if abs(self.reward_per_level[current_level - 1]) >= self.search_lims[current_level - 1]:
                print('SearchLimitBreached')
                if d:
                    self.reset_reward[current_level - 1] = 5
                    self.reset_reward[current_level] = 5
                else:
                    self.reset_reward[current_level - 1] = 2
                    self.reset_reward[current_level] = 2

                self.expected_level = np.maximum(self.expected_level - 1,0)
                # self.reward_per_level[current_level] = 0
                # self.reward_per_level[self.current_level] = 0

                # print('limit breached', current_level - 1)
                # self.tasks_bools[current_level - 1] = 1

                current_level = current_level - 1
                if current_level!=0:
                    self.check_if_search_limit_breached( current_level,d)
            # else:
            #     self.expected_level=current_level
            #     print('EL5', self.expected_level)


    def checks(self,current_level,new_state,old_state,d):

        for l in range(current_level, 0, -1):
            self.check_if_search_limit_breached( l,d)
    
    def get_possible_actions(self,current_loc,current_level):

        super_goals=copy.copy(self.get_super_manager_1(self.goal_init_state))
        current_man=copy.copy(self.get_super_manager_1(current_loc))
        if current_level!=self.n_layers:
            lim = self.lims[current_level][0]
            if current_level==0:
                self.possible_actions=[4]
            else:
                current_loc = current_man[current_level]
                if self.search_clause:
                    if super_goals[current_level]==current_man[current_level]:
                        self.possible_actions= [4]
                    else:
                        self.possible_actions= []
                        if current_loc[0] - 1 >= 0:
                            self.possible_actions.append(0)
                        if current_loc[0] + 1 <= lim - 1:
                            self.possible_actions.append(1)
                        if current_loc[1] + 1 <= lim - 1:
                            self.possible_actions.append(2)
                        if current_loc[1] - 1 >= 0:
                            self.possible_actions.append(3)

                else:
                    self.possible_actions = []
                    if current_loc[0] - 1 >= 0:
                        self.possible_actions.append(0)
                    if current_loc[0] + 1 <= lim - 1:
                        self.possible_actions.append(1)
                    if current_loc[1] + 1 <= lim - 1:
                        self.possible_actions.append(2)
                    if current_loc[1] - 1 >= 0:
                        self.possible_actions.append(3)
                    self.possible_actions.append(4)


        else:
            # self.possible_actions=[]
            # if current_loc[0]-1>=0:
            #     self.possible_actions.append(0)
            # if current_loc[0]+1<= self.maze.shape[0] - 1:
            #     self.possible_actions.append(1)
            # if current_loc[1]+1<=  self.maze.shape[0] - 1:
            #     self.possible_actions.append(2)
            # if current_loc[1]-1>=0:
            #     self.possible_actions.append(3)
            self.possible_actions = [0,1,2,3]
        return self.possible_actions

            
            
    def step(self, action,steps):
        current_loc = copy.copy(self.loc)
        current_level = copy.copy(self.current_level)
        if current_level == 0:
            #             action=5
            self.hierarchy_actions[current_level] = 4
            self.current_level = current_level + 1
        elif current_level == self.n_layers:
            assert action<=3
            #         0,1,2,3,4 --> NSEW*
            if action == 0:
                new_row = int(self.loc[0] - 1)
                new_col = int(self.loc[1])
                if new_row >= 0:
                    if self.maze[new_row][new_col] != 1:
                        self.loc = [new_row, new_col]
                    else:
                        self.reset_reward[current_level] = 0
                else:
                    self.reset_reward[current_level]=0

            if action == 1:
                new_row = int(self.loc[0] + 1)
                new_col = int(self.loc[1])
                if new_row <= self.maze.shape[0] - 1:
                    if self.maze[new_row][new_col] != 1:
                        self.loc = [new_row, new_col]
                    else:
                        self.reset_reward[current_level] = 0
                else:
                    self.reset_reward[current_level]=0

            if action == 2:
                new_row = int(self.loc[0])
                new_col = int(self.loc[1] + 1)
                if new_col <= self.maze.shape[0] - 1:
                    if self.maze[new_row][new_col] != 1:
                        self.loc = [new_row, new_col]

                    else:
                        self.reset_reward[current_level] = 0
                else:
                    self.reset_reward[current_level]=0

            if action == 3:
                new_row = int(self.loc[0])
                new_col = int(self.loc[1] - 1)
                if new_col >= 0:
                    if self.maze[new_row][new_col] != 1:
                        self.loc = [new_row, new_col]
                    else:
                        self.reset_reward[current_level] = 0
                else:
                    self.reset_reward[current_level]=0
        else:

            # self.hierarchy_actions[current_level] = action
            current_locs = self.get_super_manager_1(self.loc)[current_level]
            lim = self.lims[current_level][0]
            if action == 0:
                new_row = int(current_locs[0] - 1)
                new_col = int(current_locs[1])
                if new_row >= 0:
                    self.current_tasks_loc[current_level] = [new_row, new_col]
                    # indicates new task set
                    self.tasks_bools[current_level] = 0
                    self.hierarchy_actions[current_level] = action
                    self.current_level = current_level + 1
                else:
                    self.reset_reward[current_level]=0

            if action == 1:
                new_row = int(current_locs[0] + 1)
                new_col = int(current_locs[1])
                if new_row < lim:
                    self.current_tasks_loc[current_level] = [new_row, new_col]
                    self.tasks_bools[current_level] = 0
                    self.hierarchy_actions[current_level] = action
                    self.current_level = current_level + 1
                else:
                    self.reset_reward[current_level]=0
            if action == 2:
                new_row = int(current_locs[0])
                new_col = int(current_locs[1] + 1)
                if new_col < lim:
                    self.current_tasks_loc[current_level] = [new_row, new_col]
                    self.tasks_bools[current_level] = 0
                    self.hierarchy_actions[current_level] = action
                    self.current_level = current_level + 1
                else:
                    self.reset_reward[current_level]=0
            if action == 3:
                new_row = int(current_locs[0])
                new_col = int(current_locs[1] - 1)
                if new_col >= 0:
                    self.current_tasks_loc[current_level] = [new_row, new_col]
                    self.tasks_bools[current_level] = 0
                    self.hierarchy_actions[current_level] = action
                    self.current_level = current_level + 1
                else:
                    self.reset_reward[current_level]=0

            if action == 4:
                new_row = int(current_locs[0])
                new_col = int(current_locs[1] )
                self.current_tasks_loc[current_level] = [new_row, new_col]
                self.hierarchy_actions[current_level] = action
                self.current_level = current_level + 1
        if self.goal_init_state == self.loc:
            done = True
        else:
            done = False
        if done==True:
            self.reset_reward=[5 for x in self.reset_reward]
        if self.current_level!=self.n_layers:
            self.current_state=self.get_super_manager_1(self.loc)[self.current_level]
        else:
            self.current_state=self.loc
        return self.loc, self.current_level,self.current_state, done, {}

    def reset(self):
        self.initiate_states()
        self.loc = self.agent_init_state
        self.init_super_manager()
        self.reward_per_level = [0 for x in self.reward_per_level]
        self.reset_rewards_after_learning([0,0,0])
        self.steps_max=0
        self.current_level = 0
        self.current_tasks_loc = copy.copy(self.super_managers)
        self.tasks = [4 for x in self.current_tasks_loc]
        self.hierarchy_actions = [4 for x in range(int(self.n_layers))]
        #         need to keep track if we are in the right location according to our super manager
        self.tasks_bools = np.ones(len(self.current_tasks_loc))
        self.reset_reward = [0 for x in range(int(self.n_layers + 1))]
        self.reward_per_level = [0 for x in range(int(self.n_layers + 1))]
        self.reset_reward = [0 for x in range(int(self.n_layers + 1))]
        return self.loc

    #         reset state to something
    #         return None
    def get_reward(self,steps):

        done = False
        reward_dict={}
        if self.reward_structure==1:
            reward_dict['DoneAndSearch']=0
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -10
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 0
            reward_dict['Wall'] = -1
        elif self.reward_structure==2:
            reward_dict['DoneAndSearch'] = 1
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -10
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 0
            reward_dict['Wall'] = -1
        elif self.reward_structure==3:
            reward_dict['DoneAndSearch'] = 100
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -10
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 10
            reward_dict['Wall'] = -1



        if self.current_level == self.n_layers:
            # if find goal but manager didn't say search don't reward
            if self.loc == self.goal_init_state:
                for i in range(int(self.n_layers)):
                    x = self.reward_per_level[i]
                    if self.hierarchy_actions[i]==4:
                        # if self.tasks_bools[i]!=1:


                            self.reward_per_level[i] =  reward_dict['DoneAndSearch']
                                                        # /max(1,np.abs(x))
                            self.reward_per_level[-1] = reward_dict['DoneAndSearch']

                            # if i==1:
                            #     print('Searching and finding ')
                            #     print('Level:', i)
                            #     print('Rewards',self.reward_per_level)
                            #     print('Tasks',self.hierarchy_actions)

                    else:
                        x = self.reward_per_level[i]
                        self.reward_per_level[i] =  reward_dict['DoneAndNoSearch']
                        self.reward_per_level[-1] = reward_dict['DoneAndNoSearch']
                        # print('Not Searching and finding ')
                        # print('Level:', i)
                        # print('Rewards', self.reward_per_level)
                        # print('Tasks', self.hierarchy_actions)



                        # self.maze.shape[0]
                print('Goal Found')
                done = True

            else:

                if steps>0:
                    # print(self.reset_reward)
                    # print('....')
                    for i in range(int(self.n_layers)):

                        if self.reset_reward[i]==0:
                            x=self.reward_per_level[i]
                            self.reward_per_level[i] = x-1
                            # '/max(np.abs(x),1)
                            self.reward_per_level[-1]=-1
                        # if task satisfied 0

                        elif self.reset_reward[i]==1:
                            x = self.reward_per_level[i]
                            self.reward_per_level[i] =  x+reward_dict['TaskSatisfied']
                                                        # +self.maze.shape[0]/4
                            self.reward_per_level[-1] = reward_dict['TaskSatisfied']


                        elif self.reset_reward[i]==2:
                            x = self.reward_per_level[i]
                            self.reward_per_level[i] =  x +reward_dict['SearchLimit']
                                                        # /max(np.abs(x),1)
                            self.reward_per_level[-1] = +reward_dict['SearchLimit']
                        # if manager changed wrong  -10

                        elif self.reset_reward[i] == 3:
                            x = self.reward_per_level[i]
                            self.reward_per_level[i] =  x + reward_dict['TaskFailed']
                                                        # /max(np.abs(x),1)
                            self.reward_per_level[-1] = +reward_dict['TaskFailed']
                                                            # self.maze.shape[0]/4
                        # if wall

                        elif self.reset_reward[i] == 4:
                            x = self.reward_per_level[i]
                            self.reward_per_level[i] =  x + reward_dict['Wall']
                                                        # /max(np.abs(x),1)
                            self.reward_per_level[-1] = +reward_dict['Wall']
                        self.reset_reward[i]=0
                    self.reset_reward[-1] = 0
        # else:
        #     reward=self.reward_per_level
        #     reward[-1]=0
        reward = self.reward_per_level
        # if reward[1] > 0:
        #     print('ere')
        return reward, done

    def get_super_manager_1(self, loc):
        #         find which super manager per finest location state
        super_managers = []
        number_of_levels = int(self.n_layers)
        #         super_managers.append([np.floor(x/(2**(number_of_levels-1)/2)) for x in current_state])
        # print(loc)
        super_managers.append([np.floor(x / self.manager_view) for x in loc])
        if number_of_levels - 1 > 1:
            for i in range(number_of_levels - 1):
                super_managers.append([np.floor(x / self.manager_view) for x in super_managers[-1]])
        #                 print(i)
        else:
            super_managers.append([0, 0])
        return super_managers[::-1]

    def generate_maze(self, dim_maze=8, opaque=True):
        self.maze=np.zeros((dim_maze,dim_maze))
