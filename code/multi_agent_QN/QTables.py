


import numpy as np
import pandas as pd
import pickle
from store import store


class QLearningTable1(store):
    def __init__(self, actions, state):
        self.actions = actions  # a list
        
        if state == "RAW": 
            
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        elif state == "MAP":
           
            f = open(store.FILEPATH + 'q_table1.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()
        elif state == "PATH":
            #The q_table after path training
            f = open(store.FILEPATH + 'path_qtable1.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()

class QLearningTable2(store):
    def __init__(self, actions, state):
        self.actions = actions  

        if state == "RAW": 
            #The q_table without previous knowledge
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        elif state == "MAP":
            #The q_table after map training
            f = open(store.FILEPATH + 'q_table2.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()
        elif state == "PATH":
            #The q_table after path training
            f = open(store.FILEPATH + 'path_qtable2.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()


            
           
class ReturnQLearningTable1(store):
    def __init__(self, actions, state):
        self.actions = actions  # a list
        if state == "RAW": 
           
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        elif state == "MAP":
            #The q_table after map training
            f = open(store.FILEPATH + 'Return_q_table1.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()
        elif state == "PATH":
            #The q_table after path training
            f = open(store.FILEPATH + 'Return_q_table1.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()

class ReturnQLearningTable2(store):
    def __init__(self, actions, state):
        self.actions = actions  # a list
        if state == "RAW": 
            #The q_table without previous knowledge
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        elif state == "MAP":
            #The q_table after map training
            f = open(store.FILEPATH + 'Return_table2.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()
        elif state == "PATH":
            #The q_table after path training
            f = open(store.FILEPATH + 'Return_qtable2.txt', 'rb')
            self.q_table = pickle.load(f)
            f.close()
