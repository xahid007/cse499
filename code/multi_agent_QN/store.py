

import numpy as np
import pandas as pd


class store():
    
    FILEPATH = '/Users/peash/desktop/code/multi_agent_QN/'
    def choose_action(self, observation, epsilon):
        self.verify(observation)
        # action selection
        if np.random.uniform() < epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, alpha, gamma):
        self.verify(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + gamma * self.q_table.loc[s_, :].max()  
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += alpha * (q_target - q_predict)  # update
        

    def verify(self, state):
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )