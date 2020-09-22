import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD
import talib
import traceback

from copy import deepcopy

df = pd.read_csv('out.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.set_index("time")

class ExperienceReplay(object):
    '''This class gathers and delivers the experience'''
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, state, action, reward):
        self.memory.append([state, action, reward])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        # print(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        # for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
        for i in range(len_memory):
            # state_t, action_t, reward_t = self.memory[idx]
            state_t, action_t, reward_t = self.memory[i]
            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            if reward_t > 0:
                targets[i, action_t] += reward_t
            elif reward_t < 0:
                targets[i, action_t] += reward_t
                targets[i, 0] += reward_t
            else:
                targets[i, 0] += 1
        return inputs, targets
		
class Game(object):
    '''This is the game. It starts, then takes an action (buy or sell) at some point and finally the reverse
    action, at which point it is game over. This is where the reward is given. The state consists of a vector
    with different bar sizes for OLHC. They are just concatenated. 
    lkbk: determines how many bars to use - larger lkbk - bigger state
    '''
    def __init__(self, df, lkbk=20, max_game_len=1000, run_mode='sequential', init_idx=None):
        self.df = df
        self.lkbk = lkbk
        self.max_game_len = max_game_len
        
        self.is_over = False
        self.reward = 0
        self.run_mode =  run_mode
        self.pnl_sum = 0
        if run_mode == 'sequential' and init_idx == None:
            print('------No init_idx set for "sequential": stopping------')
            return
        else:
            self.init_idx = init_idx
        self.reset()
        
    def _update_state(self, action):
        
        '''Here we update our state'''
        self.curr_idx += 1
        self.curr_time = self.df.index[self.curr_idx]
        # print(self.curr_time)
        self.curr_price = self.df['close'][self.curr_idx]
        self.pnl = (-self.entry + self.curr_price)*self.position/self.entry
        self._assemble_state()
        _k = list(map(float,str(self.curr_time.time()).split(':')[:2]))
        self._time_of_day = (_k[0]*60 + _k[1])/(24*60) 
        self._day_of_week  = self.curr_time.weekday()/6
        self.norm_epoch = (df.index[self.curr_idx]-df.index[0]).total_seconds()/self.t_in_secs
        
        
        '''This is where we define our policy and update our position'''
        if action == 0:  
            pass
        
        elif action == 2:
            if self.position == -1:
                self.is_over = True
                self._get_reward()
                self.trade_len = self.curr_idx - self.start_idx
   
            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
            else: 
                pass
            
        elif action == 1:
            if self.position == 1:
                self.is_over = True
                self._get_reward()
                self.trade_len = self.curr_idx - self.start_idx

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
            else:
                pass
        
    
    def _assemble_state(self):
        '''Here we can add other things such as indicators and times'''
        self._get_last_N_timebars()

        state = []
        bar = self.last5m
        for col in ['prediction', 'probability']:
            state += (list(np.asarray(bar[col]))[-3:])

        self.state = np.array([])
        self.state = np.append(self.state,state)
        # self.state = np.append(self.state,self.position)
        # np.append(self.state,np.sign(self.pnl_sum))
        # self.state = np.append(self.state,self._time_of_day)
        self.state = np.append(self.state,self._day_of_week)
        
        # if np.std(self.state)!=0:
            # self.state = (np.array(self.state)-np.mean(self.state))/np.std(self.state)
        


        
        
    def _get_last_N_timebars(self):
        '''The lengths of the time windows are currently hardcoded.'''
        wdw1h = np.ceil(self.lkbk*4)
        
        self.curr_idx += 1
        self.curr_time = self.df.index[self.curr_idx]
        
        self.last5m = self.df[self.curr_time-timedelta(wdw1h):self.curr_time].iloc[-self.lkbk:]

        self.curr_idx -= 1
        self.curr_time = self.df.index[self.curr_idx]
        
        '''Making sure that window lengths are sufficient'''
        try:
            assert(len(self.last5m)==self.lkbk)
        except:
            print('****Window length too short****')
            print(len(self.last5m))
            return
            if self.run_mode == 'sequential':
                self.init_idx = self.curr_idx
                self.reset()
            else:
                self.reset()


    def _get_reward(self):
        if self.position == 1 and self.is_over:
            pnl = (self.curr_price - self.entry)/self.entry
            self.reward = np.sign(pnl)#-(self.curr_idx - self.start_idx)/1000.
        elif self.position == -1 and self.is_over:
            pnl = (-self.curr_price + self.entry)/self.entry
            self.reward = np.sign(pnl)#-(self.curr_idx - self.start_idx)/1000.
        return self.reward
            
    def observe(self):
        return np.array([self.state])

    def act(self, action):
        self._update_state(action)
        reward = self.reward
        game_over = self.is_over
        return self.observe(), reward, game_over

    def reset(self):
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0
        
        if self.run_mode == 'random':
            self.curr_idx = np.random.randint(0,len(df)-3000)
            
        elif self.run_mode == 'sequential':
            self.curr_idx = self.init_idx
            
        self.t_in_secs = (df.index[-1]-df.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.df.index[self.curr_idx]
        self._get_last_N_timebars()
        self.state = []
        self.position = 0
        self._update_state(0)
		
def run(df,fname):
    # parameters
    epsilon_0 = .001
    num_actions = 3 
    epoch = 5000
    max_memory = 100
    
    batch_size = 500
    lkbk = 3
    START_IDX = 300

    env = Game(df, lkbk=lkbk, max_game_len=1000,init_idx=START_IDX,run_mode='sequential')
    hidden_size = num_actions*len(env.state)*2
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(len(env.state),), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(SGD(lr=.05), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("indicator_model.h5")

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    loss_cnt = 0
    wins = []
    losses = []
    pnls = []
    for e in range(epoch):
        action = 0
        # epsilon = epsilon_0**(np.log10(e))
        epsilon = 0.4
        env = Game(df, lkbk=lkbk, max_game_len=1000,init_idx=env.curr_idx,run_mode='sequential')
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        cnt = 0
        while not game_over:
            print(cnt)
            cnt += 1
            input_tm1 = input_t
            # get next action

            if env.position:
                print('***Time Exit***')
                action = exit_action

            elif np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)[0]
                if env.position == 0:
                    if action != 0:
                        print('***random entry***')
                        input_state_start = deepcopy(input_tm1)
                        action_start = action
                    if action == 2:
                        exit_action = 1
                    elif action == 1:
                        exit_action = 2
                    
            elif env.position == 0:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
                if action:
                    input_state_start = deepcopy(input_tm1)
                    action_start = action
                    exit_action = np.argmin(q[0][1:])+1
				

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward > 0:
                win_cnt += 1
            elif reward < 0:
                loss_cnt += 1

            # store experience
            # if action or len(exp_replay.memory)<20 or np.random.rand() < 0.1:
                # exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            env.pnl_sum = sum(pnls)

            # zz = model.train_on_batch(inputs, targets)
            # loss += zz
        exp_replay.remember(input_state_start, action_start, reward)
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
        loss = model.train_on_batch(inputs, targets)
        prt_str = ("Epoch {:03d} | Loss {:.2f} | pos {} | len {} | sum pnl {:.2f}% @ {:.2f}% | eps {:,.4f} | {} | entry price {} | current price {}".format(e, 
                                                                                      loss, 
                                                                                      env.position, 
                                                                                      env.trade_len,
                                                                                      sum(pnls)*100,
                                                                                      env.pnl*100,
                                                                                      epsilon,
                                                                                      env.curr_time, env.entry, env.curr_price))
        print(prt_str)

        fid = open(fname,'a')
        fid.write(prt_str+'\n')
        fid.close()
        pnls.append(env.pnl)
        if not e%10:
            print('----saving weights-----')
            model.save_weights("indicator_model.h5", overwrite=True)
			
fname = 'output1.dat'
fid = open(fname,'w')
fid.close()
run(df,fname)