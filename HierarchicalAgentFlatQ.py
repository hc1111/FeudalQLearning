import numpy as np
import copy as copy
import random as random
# Qtable to keep track of state action values
class HQT:
    def __init__(self, env,step_size):
        self.counter=0
        self.Q_table = {}
        self.Visit_table = {}
        # used for learning rate
        self.T=75
        self.layer_states = {}
        #  Create Dictionary for Q values
        for level in range(int(env.n_layers + 1)):
            ns = (env.maze.shape[0] / (step_size ** (env.n_layers - level))) ** 2
            self.Q_table[level] = {}
            self.Visit_table[level] = {}
            self.layer_states[level] = ns ** 0.5
            for state in range(int(ns)):
                self.Visit_table[level][state] = {}
                self.Q_table[level][state] = {}
                for task in range(env.na + 1):
                    self.Visit_table[level][state][task] = {}
                    self.Q_table[level][state][task] = {}
                    if level != env.n_layers:
                        for action in range(env.na + 1):
                            self.Visit_table[level][state][task][action] = 0
                            self.Q_table[level][state][task][action] = 0
                    elif level > 0:
                        for action in range(env.na):
                            self.Visit_table[level][state][task][action] = 0
                            self.Q_table[level][state][task][action] = 0
                    else:
                        for action in [4]:
                            self.Visit_table[level][state][task][action] = 0
                            self.Q_table[level][state][task][action] = 0

    def get_actions_possible(self, env, level):
        self.actions_possible = []
        if level == 0:
            self.actions_possible=[4]
        elif level == env.n_layers:
            self.actions_possible=[0, 1, 2, 3]
        else:

            self.actions_possible=[0,1,2,3,4]


        return self.actions_possible

    def choose_action(self, env, level,episodes):
        #  get possible actions
        allowed_actions = env.possible_actions


        if level == 0:
            # only one possible action --> Search
            return 4

        elif level != env.n_layers:
            #  Get task from environment
            task = env.hierarchy_actions[level - 1]
            #  Get state value
            state = env.get_super_manager_1(env.loc)[level]
        else:
            self.counter=self.counter+1
            #  Get task
            task = env.hierarchy_actions[level - 1]
            state = env.loc

        #  Define state from x,y coordinates
        state1 = state[0] * self.layer_states[level] + state[1]
        #  Define action values
        action_values = [self.Q_table[level][state1][task][y] for y in allowed_actions]
        #  Get temperature
        Beta=0.992
        Tmin=0.5
        self.T=Tmin+Beta*(self.T-Tmin)
        #  Random Action
        self.epsilon=0.10
        #  if only one choice possible e.g. manager with goal in domain can only choose search
        if len(action_values)==1:
            boltz= [1]
        else:
            # boltzmanm distribution with temperature
            #  Subtract max value to ensure no crazy numbers
            boltz = [np.exp((x)/self.T) for x in [action_values-np.max(action_values)]][0]
            boltz = [x for x in boltz / np.sum(boltz)]
            # print(boltz)
        try:
            #  Epsilon Greedy addition
            if random.random() < self.epsilon:
                choice_action = random.choice(allowed_actions)
            else:
                choice_action=np.random.choice(allowed_actions,p=boltz)
        except:
            print('dsd')




        return choice_action



    def update_Q_values(self, env, ns,state, level, reward, action, task, done,episode,num_episodes,discount_factor,lr,state_1,ns_1):
        tau=50
        n=episode/20
        #  learning ratea
        self.alpha_new=np.maximum((0.5*tau)/(tau+n),0.1)
        # old state and new state from x,y coordinates
        state1 = state[0] * self.layer_states[level] + state[1]
        n_state1 = ns[0] * self.layer_states[level] + ns[1]

        #  get allowed actions for new state
        allowed_actions=env.get_possible_actions(ns_1,level)
        # current q value before update
        old_q_value=copy.copy(self.Q_table[level][state1][task][action])
        #  cgeck if non primitive
        if level!=env.n_layers:
            #  cgeck not super manager
            if level!=0:
                #  find where the goal is on this level
                goal_level_1 = env.get_super_manager_1(env.goal_init_state)[level]
                #  if not in goal domain
                if ns!=goal_level_1:
                    # if np.sum([x for x in self.Q_table[level][n_state1][task].values()][:4])==0:
                    max_future=np.max([x[1] for x in enumerate(self.Q_table[level][n_state1][task].values()) if x[0] in allowed_actions])
                else:
                    #  can only use search action
                    max_future=np.max([x for x in self.Q_table[level][n_state1][task].values()][4])
            else:
                #  can only use super manager allowed actions
                max_future = np.max([x[1] for x in enumerate(self.Q_table[level][n_state1][task].values()) if x[0] in allowed_actions])
        # if primitive
        else:

            max_future = np.max([x[1] for x in enumerate(self.Q_table[level][n_state1][task].values()) if x[0] in allowed_actions])
        # if level!=env.n_layers:
        #     if level!=0:
        #         if state==env.get_super_manager_1(env.goal_init_state)[1]:
        #             if action!=4:
        #                 print('hold up dodginess at large')
        #         else:
        #             if state!=env.get_super_manager_1(env.goal_init_state)[1]:
        #                 if action==4:
        #                     print('hold up dodginess at large')
        # if level==env.n_layers:
        #     goal_level_1 = env.get_super_manager_1(env.goal_init_state)[1]
        #     act_level_1 = env.get_super_manager_1(state_1)[1]
        #     if goal_level_1==act_level_1:
        #         if task!=4:
        #             print('hold up dodginess at large')




        #  Q value = old Q value + learning rate*(Reward + DF*(Max Value Action from next state)- old Q Value)
        # if not done:
        current_q_value =self.Q_table[level][state1][task][action]+self.alpha_new*(reward +discount_factor*max_future-self.Q_table[level][state1][task][action])
        # else:
            # current_q_value = rewa/rd
        # update dictionaries
        self.Q_table[level][state1][task][action] = current_q_value
        self.Visit_table[level][state1][task][action] += 1
        #  return values
        return old_q_value,current_q_value,max_future



