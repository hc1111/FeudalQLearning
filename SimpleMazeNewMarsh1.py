import numpy as np
import copy
import random
import clingo

class Hierarchical_Maze():
    def __init__(self,shape,manager_layers,manager_view,reward_structure,search_clause,opaque,block_dim,recursion,chance):

        if manager_view**(manager_layers-1)>shape:
            print('View is too large fr number of layers or vice versa')
            assert manager_view**(manager_layers-1)<=shape


        self.dim=shape
        self.opaque = opaque
        self.block_dim = block_dim
        self.search_clause=search_clause
        self.chance=chance
        self.recursion=recursion
        if recursion:
            self.generate_maze(opaque)
        else:
            self.generate_maze_2()
        self.init_goal()
        self.init_state()
        self.manager_view=manager_view
        self.n_layers=manager_layers
        self.lims=self.get_super_manager_1([shape, shape])
        self.rewards=[0 for x in range(manager_layers+1)]
        self.steps = [0 for x in range(self.n_layers + 1)]

        self.reset_reward=[0 for x in range(manager_layers+1)]
        self.reset_reward_2 = [0 for x in range(manager_layers + 1)]
        self.reward_structure=reward_structure
        self.na=4
        self.count_level_change={}
        for l in range(self.n_layers+1):
            self.count_level_change[l]={'TS':0,'TF':0,'SL':0}
    def reset(self,g=None,agent_loc=None):
        if g :
            self.init_goal()
        if agent_loc==None:
            self.init_state()
        else:
            self.agent_init_state=agent_loc
            assert self.agent_init_state!=self.goal_init_state

        self.rewards = [0 for x in range(self.n_layers + 1)]
        self.steps = [0 for x in range(self.n_layers + 1)]
        self.reset_reward=[0 for x in range(self.n_layers+1)]
        self.reset_reward_2 = [0 for x in range(self.n_layers + 1)]
        # self.count_level_change={}
        self.count_level_change = {}
        for l in range(self.n_layers + 1):
            self.count_level_change[l] = {'TS': 0, 'TF': 0, 'SL': 0}

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

    def generate_maze_2(self):
        self.maze=np.zeros((self.dim,self.dim))
        for x1 in range(self.dim):
            for  y1 in range(self.dim):
                choice = np.random.uniform(0,1,1)

                if choice <self.chance:
                    self.maze[x1,y1]=1
    def generate_maze(self,opaque):
        base_cell = self.create_good_cell(self.block_dim)
        if opaque:
            bad_base_cell = np.ones(base_cell.shape)
            self.bad_cell = bad_base_cell
        else:
            bad_base_cell = self.create_bad_cell(self.block_dim)

        good_cell_level_block = copy.copy(base_cell)
        bad_cell_level_block = copy.copy(bad_base_cell)
        level2 = copy.copy(base_cell)
        while level2.shape[0] < self.dim:
            good_cell_level_block_new = self.build_good_block(base_cell, good_cell_level_block, bad_cell_level_block)
            level2 = good_cell_level_block
            bad_cell_level_block_new = self.build_bad_block(bad_base_cell, good_cell_level_block, bad_cell_level_block)
            good_cell_level_block = copy.copy(good_cell_level_block_new)
            bad_cell_level_block = copy.copy(bad_cell_level_block_new)
        self.maze = level2




    def init_goal(self):
        empty_cells = [[x, y] for x, y in zip(np.where(self.maze == 0)[0], np.where(self.maze == 0)[1])]
        self.goal_init_state = random.choice(empty_cells)

    def init_state(self):
        # if not self.recursion:
        empty_cells = [[x, y] for x, y in zip(np.where(self.maze != 990)[0], np.where(self.maze != 990)[1])]
        # else:
        #     empty_cells = [[x, y] for x, y in zip(range(self.maze.shape[0]), range(self.maze.shape[0]))]
        empty_cells = [x for x in empty_cells if x not in [self.goal_init_state]]
        # print(empty_cells)
        self.agent_init_state = random.choice(empty_cells)

    def step(self,action,level,HQT_new):
        loc=copy.copy(HQT_new.loc)

        # possible_actions=self.possible_actions(level,loc)
        # assert action in possible_actions
        if level!=self.n_layers:
            loc1=self.get_super_manager_1(loc)[level]
        if level==0:
            assert action==4
            HQT_new.tasks[level]=action
            HQT_new.current_level=level+1
        elif level==self.n_layers:

            for i in range(level):
                self.steps[i] += 1
            assert action<=3
            # NSEW
            if action==0:
                new_y=int(loc[0]-1)
                new_x=int(loc[1])
                if new_y >= 0:
                    loc = [new_y, new_x]
                    if self.maze[new_y][new_x] == 1:
                        self.reset_reward_2[level] = 4
                    else:
                        self.reset_reward_2[level] = 0
                    # self.reset_reward[level]=0
                else:
                    new_y = int(loc[0])
                    new_x = int(loc[1])
                    if self.maze[new_y][new_x] == 1:
                        self.reset_reward_2[level] = 4
                    else:
                        self.reset_reward_2[level] = 0
            elif action==1:
                new_y=int(loc[0]+1)
                new_x=int(loc[1])
                if new_y <= self.dim-1:
                    loc = [new_y, new_x]

            elif action==2:
                new_y=int(loc[0])
                new_x=int(loc[1]+1)
                if new_x <= self.dim-1:
                    loc = [new_y, new_x]


            elif action==3:
                new_y=int(loc[0])
                new_x=int(loc[1]-1)
                if new_x >= 0:
                    loc = [new_y, new_x]
            new_y = int(loc[0])
            new_x = int(loc[1])
            if self.maze[new_y][new_x] == 1:
                self.reset_reward_2[level] = 4
            else:
                self.reset_reward_2[level] = 0

        else:
            lim=self.lims[level][0]
            if action==0:
                new_y = int(loc1[0] -1)
                new_x = int(loc1[1])
                # if new_y >= 0:

                # else:
                #     self.reset_reward[level]=0


            if action==1:
                new_y = int(loc1[0] + 1)
                new_x = int(loc1[1])


            if action==2:
                new_y = int(loc1[0] )
                new_x = int(loc1[1]+1)
            if action==3:
                new_y = int(loc1[0] )
                new_x = int(loc1[1]-1)


            if action==4:

                HQT_new.current_tasks_loc[level]=loc1

            else:
                HQT_new.current_tasks_loc[level] = [new_y, new_x]
                HQT_new.tasks_bools[level] = 0



            HQT_new.tasks[level] = action
            # print(level)
            HQT_new.hierarchy_actions[level] = action

            HQT_new.current_level = level + 1
        # assert HQT_new.hierarchy_actions[level].isin([0,1,2,3,4])
        if self.goal_init_state==loc:
            done=True
        else:
            done=False

        if done:
            self.reset_reward=[5 for x in self.reset_reward ]

        return loc,done


    def possible_actions(self,level,loc):
        super_goals=copy.copy(self.get_super_manager_1(self.goal_init_state))
        if level!=self.n_layers:
            loc=self.get_super_manager_1(loc)[level]
        if level==0:
            possible_actions=[4]
        elif level==self.n_layers:
            possible_actions=[0,1,2,3]
        else:
            lim = self.lims[level][0]
            if super_goals[level]==loc:
                possible_actions=[4]
            else:
                if self.search_clause:
                    possible_actions = [0,1,2,3]

                else:
                    possible_actions=[0,1,2,3]

        return possible_actions


    def get_reward_dict(self):
        reward_dict={}
        if self.reward_structure==1:
            reward_dict['DoneAndSearch']=0
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -1*(2*self.manager_view**2)
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 0
            reward_dict['Wall'] = -2*(self.manager_view)
        elif self.reward_structure==2:
            reward_dict['DoneAndSearch'] = 0
            reward_dict['DoneAndNoSearch'] = 0
                                             # -10*(self.block_dim*2-2)
            reward_dict['TaskFailed'] = -10*(self.block_dim*2-2)
            # self.maze.shape[0]*3
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 0
            reward_dict['Wall'] = -2*(self.block_dim*2-2)
            # self.maze.shape[0]
        elif self.reward_structure==3:
            reward_dict['DoneAndSearch'] = 0
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -40
            reward_dict['SearchLimit'] = -1
            reward_dict['TaskSatisfied'] = 5
            reward_dict['Wall'] = -3
        elif self.reward_structure==4:
            reward_dict['DoneAndSearch'] = 0
            reward_dict['DoneAndNoSearch'] = 0
            reward_dict['TaskFailed'] = -.1 * (2 * self.manager_view ** 2)
            reward_dict['SearchLimit'] = -.1
            reward_dict['TaskSatisfied'] = 0
            reward_dict['Wall'] = -.2 * (self.manager_view)

        return reward_dict

    def reward(self, HQT_new, done,old_level):
        reward_dict = self.get_reward_dict()
        level = copy.copy(HQT_new.current_level)
        if old_level == self.n_layers:
            if done:
                for i in range(int(self.n_layers)):

                    if self.reset_reward_2[- 1] == 4:
                        # self.rewards[i] += - 1
                        r_lower = reward_dict['Wall']


                    else:
                        # self.reset_reward_2[- 1] == 4:
                        # self.rewards[i] += reward_dict['Wall']
                        r_lower = reward_dict['SearchLimit']


                    r = self.rewards[i]

                    if HQT_new.tasks[level - 1] == 4:
                        self.rewards[i] = 0
                        # r+r_lower
                            # r+r_lower
                        self.rewards[-1] = r_lower
                            # r_lower+reward_dict['DoneAndSearch']
                    else:
                        self.rewards[i] = 0
                            # r+r_lower
                        # r+r_lower
                        self.rewards[-1] = r_lower+reward_dict['DoneAndNoSearch']
                        # r_lower+reward_dict['DoneAndNoSearch']

            else:
                for i in range(int(self.n_layers)):

                    cm=self.get_super_manager_1(HQT_new.loc)[i]
                    gm=self.get_super_manager_1(self.goal_init_state)[i]
                    r = self.rewards[i]
                    # if i==self.n_layers-1:
                    #     r_1 = 0
                    #     q = -1
                    #
                    #
                    #     # if super manager has said to search e.g. manager 1 to manager 2 on 3 level
                    #     #  then rewards are as before, else the penalty for getting to goal manager area exists
                    #     # i.e. like finding goal without knowing to search for it
                    #     #  i is our manager layer, i-1 is our super manager
                    #     #  q indicates the lower layer '
                    #
                    #     # if  HQT_new.tasks[i - 1] == 4:
                    #     #     r_1=self.rewards[i]
                    #     # else:
                    #     #     r_1 = self.rewards[i]+reward_dict['DoneAndNoSearch']
                    # else:
                    q = i + 1
                    r_1 = 0
                    self.steps[i] += 1

                    if self.reset_reward_2[- 1] == 4:
                        r_lower = reward_dict['Wall']
                    else:
                        r_lower = reward_dict['SearchLimit']
                    if cm==gm:
                        self.rewards[i]=0
                        if self.reset_reward[i] == 1:
                            self.rewards[q] = r_1+r_lower+  reward_dict['TaskSatisfied']

                        elif self.reset_reward[i] == 2:
                            self.rewards[q] = r_1+r_lower

                        elif self.reset_reward[i] == 3:
                            self.rewards[q] =r_1+  r_lower+ reward_dict['TaskFailed']

                        else:
                            self.rewards[q] =r_1+ r_lower


                    else:
                        if self.reset_reward[i] == 1:
                            self.rewards[i] = r + r_lower
                            self.rewards[q] = r_1+r_lower+  reward_dict['TaskSatisfied']

                        elif self.reset_reward[i] == 2:
                            self.rewards[i] = r + r_lower
                            self.rewards[q] =r_1+ r_lower

                        elif self.reset_reward[i] == 3:
                            self.rewards[i] = r + r_lower
                            self.rewards[q] = r_1+  r_lower+ reward_dict['TaskFailed']

                        else:
                            self.rewards[i] = r + r_lower
                            self.rewards[q] = r_1+r_lower


                        # reset reward reasoning
                    self.reset_reward_2[i] = 0
                    self.reset_reward[i] = 0
                self.reset_reward[-1] = 0
                self.reset_reward_2[-1] = 0



        rewards=np.array(self.rewards)/np.array([np.maximum(1,x) for x in self.steps])
        return self.rewards

    def reset_rewards_after_learning(self,old_reset):
        for i,x in enumerate(old_reset):
            if x !=0:
                self.rewards[i] = 0
                self.steps[i]=0


    def create_bad_cell(self, dim):
        k = dim

        asp_program = "#const n={}.\n".format(k)
        asp_program += """axis(0..n-1).
        grid(X,Y):-axis(X),axis(Y).
        (n*n/2)-2{wall(X,Y):grid(X,Y)}.


        3{maze(n-1,n-1);maze(0,0);maze(0,n-1);maze(n-1,0)}3.

        adjacent(X,Y,X,W):- not wall(X,W),not wall(X,Y), grid(X,Y),grid(X,W), |Y-W|=1.
        adjacent(X,Y,Z,Y):- not wall(X,Y),not wall(Z,Y), grid(X,Y),grid(Z,Y), |Z-X|=1.
        maze(X,Y) :- grid(X,Y),not wall(X,Y).
        reachable(X,Y,X,Y):- grid(X,Y),not wall(X,Y).
        reachable(X,Y,I,J):- reachable(X,Y,Z,W),adjacent(Z,W,I,J).
        :- not reachable(X,Y,Z,W), maze(X,Y),maze(Z,W).
        maze_1(X,Y,1):-maze(X,Y).
        maze_1(X,Y,0):-wall(X,Y).
        #show maze_1/3.


        """
        control = clingo.Control()
        control.add("base", [], asp_program)
        control.ground([("base", [])])
        control.configuration.solve.models = 0
        solutions = []
        with control.solve(yield_=True) as handle:
            for model in handle:
                #         solution = [0]*k
                #         print(model)
                solution = np.zeros((k, k))

                for atom in model.symbols(shown=True):
                    if atom.name == "maze_1":
                        i = atom.arguments[0].number
                        j = atom.arguments[1].number
                        k1 = atom.arguments[2].number
                        #                 print(i,j,k1)
                        #                 print(i,j,k1)
                        solution[i][j] = k1
                #             print("Solution:",solution)
                solutions.append(solution)
        rand_solution = np.random.choice(list(range(len(solutions))))
        self.bad_cell = rand_solution
        return solutions[rand_solution]

    def create_good_cell(self, dim):
        k = self.block_dim

        asp_program = "#const n={}.\n".format(k)
        asp_program += """axis(0..n-1).
        grid(X,Y):-axis(X),axis(Y).
        (n*n/2)-2{wall(X,Y):grid(X,Y)}.


        3{maze(n-1,n-1);maze(0,0);maze(0,n-1);maze(n-1,0)}3.

        adjacent(X,Y,X,W):- not wall(X,W),not wall(X,Y), grid(X,Y),grid(X,W), |Y-W|=1.
        adjacent(X,Y,Z,Y):- not wall(X,Y),not wall(Z,Y), grid(X,Y),grid(Z,Y), |Z-X|=1.
        maze(X,Y) :- grid(X,Y),not wall(X,Y).
        reachable(X,Y,X,Y):- grid(X,Y),not wall(X,Y).
        reachable(X,Y,I,J):- reachable(X,Y,Z,W),adjacent(Z,W,I,J).
        :- not reachable(X,Y,Z,W), maze(X,Y),maze(Z,W).
        maze_1(X,Y,0):-maze(X,Y).
        maze_1(X,Y,1):-wall(X,Y).
        #show maze_1/3.


        """
        control = clingo.Control()
        control.add("base", [], asp_program)
        control.ground([("base", [])])
        control.configuration.solve.models = 0
        solutions = []
        with control.solve(yield_=True) as handle:
            for model in handle:
                #         solution = [0]*k
                #         print(model)
                solution = np.zeros((k, k))

                for atom in model.symbols(shown=True):
                    if atom.name == "maze_1":
                        i = atom.arguments[0].number
                        j = atom.arguments[1].number
                        k1 = atom.arguments[2].number
                        #                 print(i,j,k1)
                        #                 print(i,j,k1)
                        solution[i][j] = k1
                #             print("Solution:",solution)
                solutions.append(solution)
        rand_solution = np.random.choice(list(range(len(solutions))))
        self.good_cell = solutions[rand_solution]
        return self.good_cell


    def build_good_block(self, base_cell_map, base_cell, bad_base_cell):
        good_cell_level_1 = {}

        for y in range(base_cell_map.shape[0]):
            #         base_cell=random.choice(basecells)
            if base_cell_map[y, 0] == 1:
                good_cell_level_1[y] = bad_base_cell

            else:
                good_cell_level_1[y] = base_cell
            for x in range(1, base_cell_map.shape[0], 1):
                if base_cell_map[y, x] == 1:
                    good_cell_level_1[y] = np.concatenate((good_cell_level_1[y], bad_base_cell), axis=1)
                else:
                    good_cell_level_1[y] = np.concatenate((good_cell_level_1[y], base_cell), axis=1)
        good_cell_level_block = good_cell_level_1[list(good_cell_level_1.keys())[0]]
        for j in list(good_cell_level_1.keys())[1:]:
            good_cell_level_block = np.concatenate((good_cell_level_block, good_cell_level_1[j]), axis=0)
        return good_cell_level_block

    def build_bad_block(self, bad_base_cell_map, base_cell, bad_base_cell):
        bad_cell_level_1 = {}

        for y in range(bad_base_cell_map.shape[0]):
            if bad_base_cell_map[y, 0] == 0:
                bad_cell_level_1[y] = base_cell

            else:
                bad_cell_level_1[y] = bad_base_cell
            for x in range(1, bad_base_cell_map.shape[0], 1):
                if bad_base_cell_map[y, x] == 0:
                    bad_cell_level_1[y] = np.concatenate((bad_cell_level_1[y], base_cell), axis=1)
                else:
                    bad_cell_level_1[y] = np.concatenate((bad_cell_level_1[y], bad_base_cell), axis=1)
        bad_cell_level_block = bad_cell_level_1[list(bad_cell_level_1.keys())[0]]
        for j in list(bad_cell_level_1.keys())[1:]:
            bad_cell_level_block = np.concatenate((bad_cell_level_block, bad_cell_level_1[j]), axis=0)
        return bad_cell_level_block





    def checks_if_level_should_change(self,old_loc,new_loc,level,done,HQT_new,mvg):

        current_managers=self.get_super_manager_1(old_loc)
        new_managers=self.get_super_manager_1(new_loc)
        # print('current managers',current_managers)
        # print('new managers',new_managers)
        # print('current tasks', HQT_new.current_tasks_loc)

        if level!=0:
            # print(0)
            # print(HQT_new.current_tasks_loc[level-1:])
            # print(new_managers[level-1:])
            if HQT_new.current_tasks_loc[level-1]==new_managers[level-1]:

                # Ensure that the goal wsan't just completed
                # print(1)
                if new_managers[level-1]!=current_managers[level-1]:
                    # print('1a')
                    # print('TB', HQT_new.tasks_bools)
                    # print('l1',level-1)
                    # print('check',HQT_new.tasks_bools[level-1])
                    # print(done)



                    if HQT_new.tasks_bools[level-1]!=1:
                        # print('2a')
                        if HQT_new.hierarchy_actions[level-1]==4:
                            if done:
                                self.reset_reward[level-1]=5
                                self.reset_reward[level]=5
                            else:
                                self.reset_reward[level-1]=0
                                self.reset_reward[level]=0
                        else:
                            self.reset_reward[level-1]=1
                            self.reset_reward[level]=1
                        HQT_new.tasks_bools[level-1]=1
                        HQT_new.expected_level=np.maximum(HQT_new.expected_level-1,0)
                        # print('Task Satisfied')
                        self.count_level_change[level]['TS'] += 1
                        level=level-1
                        self.checks_if_level_should_change(old_loc,new_loc,level,done,HQT_new,mvg)
                    else:


                        # print('2b')
                        HQT_new.expected_level=level

                else:
                    # print('1b')
                    HQT_new.expected_level=level
            # else:
                # HQT_new.current_tasks_loc[:level]==new_managers[:level]:
            elif  new_managers[level-1]!=current_managers[level-1]:
                    self.reset_reward[level-1]=3
                    self.reset_reward[level]=3
                    HQT_new.expected_level = np.maximum(HQT_new.expected_level - 1, 0)
                    # print('Task Failed')
                    self.count_level_change[level]['TF']+=1
                    # ] = {'TS': 0, 'TF': 0, 'SL': 0}
                    level=level-1
                    self.checks_if_level_should_change( old_loc, new_loc, level, done, HQT_new,mvg)
                # elif new_managers[level-1]==current_managers[level-1]:
                #     self.reset_reward[level - 1] = 3
                #     self.reset_reward[level] = 3
                #     HQT_new.expected_level = np.maximum(HQT_new.expected_level - 1, 0)
                #     # print('Task Failed')
                #     self.count_level_change[level]['TF'] += 1
                #     # ] = {'TS': 0, 'TF': 0, 'SL': 0}
                #     level = level - 1
                #     self.checks_if_level_should_change(old_loc, new_loc, level, done, HQT_new,mvg)


            # elif mvg:
            #     self.reset_reward[level-1]=3
            #     self.reset_reward[level]=3
            #     HQT_new.expected_level = np.maximum(HQT_new.expected_level - 1, 0)
            #     # print('Task Failed')
            #
            #     level=level-1
            #     self.checks_if_level_should_change( old_loc, new_loc, level, done, HQT_new,mvg)

            else:
                # print(2)

                HQT_new.expected_level = level
        for l in range(level,0,-1):
            if l-1>=0:
                if abs(self.steps[l-1])>=HQT_new.search_lims[l-1]:
                    self.reset_reward[l-1]=2
                    self.reset_reward[l]=2
                    # if mvg:
                    #     if self.reset_reward==[2,2,3]:
                    #         print('x')
                    HQT_new.expected_level = np.maximum(HQT_new.expected_level - 1, 0)
                    # print('Search Limit Breached')
                    self.count_level_change[level]['SL'] += 1
                    level=l-1
                    if level!=0:
                        self.checks_if_level_should_change( old_loc, new_loc, level, done, HQT_new,mvg)









