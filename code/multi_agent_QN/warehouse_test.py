

from new_small_maze import Maze
from QTables import QLearningTable1
from QTables import QLearningTable2

from QTables import ReturnQLearningTable1
from QTables import ReturnQLearningTable2

import matplotlib.pyplot as plt
import pickle
\
HUMANWALK1 = [4,4,4,4,4,4,4,4]
UNIT = 20
STATE0 = "RAW"
STATE1 = "MAP"
STATE2 = "PATH"

CURRENTSTATE = STATE2

RECORDDATA = False

FILEPATH = "/Users/peash/desktop/code/multi_agent_QN/"


def update():
    totalReward1 = 0
    totalReward2 = 0
    rewardList1 = []
    rewardList2 = []
    totalRewardList = []
    #check whether the agent is available to move
    freeze1 = False
    freeze2 = False
    #check whether facing fixed obstacles
    backup1 = False
    backup2 = False
    for episode in range(100):
        # initial observation
        observation1, observation2 = env.resetRobot()
       
        human1, human2 = env.resetHuman()
        humanWalkHelper = 0
        wait_time1 = 0
        wait_time2 = 0
       
        while True:
            
            env.render()
            
           
    
            # RL choose action based on observation
            if freeze1:
                action1 = 4
            else:
                # Choose action
                if not backup1:
                    action1 = chooseAction(episode, RL1, observation1)   
                    if stateChecking(human1, observation1, action1) == 'no_collision':
                        observation1_, reward1, done1 = env.step1(action1)
                    else:
                        if wait_time1 > 5:
                            chooseActionForRobots(stateChecking(human1, observation1, action1), 
                                         observation1_, reward1, done1, "robot1")
                            backup1 = True
                            wait_time1 = 0
                        else:
                            observation1_, reward1, done1 = env.step1(4)
                            wait_time1 +=1 
                    learn (episode, RL1, action1, reward1, observation1, observation1_)
                else:
                    action1 = chooseAction(episode, backupRL1, observation1)
                    observation1_, reward1, done1 = env.step1(action1, human1)
                    learn (episode, backupRL1, action1, reward1, observation1, observation1_)      
                    
                totalReward1+=reward1
                observation1 = observation1_
            
            if freeze2:
                action2 = 4
            else:
                # Choose action
                if not backup2:
                    action2 = chooseAction(episode, RL2, observation2)   
                    if stateChecking(human1, observation2, action2) == 'no_collision':
                        observation2_, reward2, done2 = env.step2(action2)
                    else:
                        if wait_time2 > 5:
                            chooseActionForRobots(stateChecking(human1, observation2, action2), 
                                          observation2_, reward2, done2, "robot2")
                            backup2 = True
                            wait_time2 = 0
                        else:
                            observation2_, reward2, done2 = env.step2(4)
                            wait_time2 +=1 
                    learn (episode, RL2, action2, reward2, observation2, observation2_)
                else:
                    action2 = chooseAction(episode, backupRL2, observation2)
                    observation2_, reward2, done2 = env.step2(action2, human1)
                    learn (episode, backupRL2, action2, reward2, observation2, observation2_)      
                    
                totalReward2+=reward2

                # swap observation
                observation2 = observation2_
                
            
            
            # break while loop when end of this episode
                
            if (done1 == 'hit' or done1 == 'arrive') and (done2 == 'hit' or done2 == 'arrive') :
                print (episode, 'trial: ','Robot1: ', totalReward1, '; Robot2: ', totalReward2)
                rewardList1.append(totalReward1)
                rewardList2.append(totalReward2)
                totalRewardList.append(totalReward1+totalReward2)
                totalReward1 = 0
                totalReward2 = 0
                
                if done1 == 'arrive' and done2 == 'arrive' :
                    for i in range(50):
                        if startReturnTable (episode, observation1, ReturnRL1,1) == 'arrive':
                            break
                    for i in range(50): 
                        if startReturnTable (episode, observation2, ReturnRL2,2) == 'arrive':
                            break
                    

                freeze1 = False
                freeze2 = False
                
                break
            if done1 == 'hit' or done1 == 'arrive':
                freeze1 = True
            if done2 == 'hit' or done2 == 'arrive':
                freeze2 = True
           
       
        # Train the map and dump into pickle
    if RECORDDATA:
        f1 = open(FILEPATH + 'Return_qtable1.txt', 'wb');
        pickle.dump(ReturnRL1.q_table,f1)
        f1.close()
        f2 = open(FILEPATH + 'Return_qtable2.txt', 'wb');
        pickle.dump(ReturnRL2.q_table,f2)
        f2.close()
        
        f4 = open(FILEPATH + 'path_qtable1.txt', 'wb');
        pickle.dump(RL1.q_table,f4)
        f4.close()
        f5 = open(FILEPATH + 'path_qtable2.txt', 'wb');
        pickle.dump(RL2.q_table,f5)
        f5.close()
        

    plot(rewardList1)
    plot(rewardList2)
      
    plot(totalRewardList)                    
           
    print('simulation over')
    env.destroy()
    #print(rewardList)
    
def startReturnTable (episode, observation, RL, robotNumber):
    #observation1, observation2, observation3 = env.resetRobot()
    while True:
        env.render()
        action = chooseNoRandomAction(RL, observation)
        if robotNumber == 1:
            observation_, reward, done = env.returnStep1(action)
        elif robotNumber == 2:
            observation_, reward, done = env.returnStep2(action)
        
        learn (episode, RL, action, reward, observation, observation_)
        observation = observation_
        #print (done)
        if done == 'arrive' or done == 'hit':
            break
    return done

def chooseActionForRobots(state, observation, reward, done, robot):

    if (robot == "robot1"):
        if state == 'upward_collision':
            observation, reward, done = env.step1(1)
        elif state == 'downward_collision':
            observation, reward, done = env.step1(0)
        elif state == 'left_collision':
            observation, reward, done = env.step1(2)
        elif state == 'right_collision':
            observation, reward, done = env.step1(3)  

    if (robot == "robot2"):
        if state == 'upward_collision':
            observation, reward, done = env.step2(1)
        elif state == 'downward_collision':
            observation, reward, done = env.step2(0)
        elif state == 'left_collision':
            observation, reward, done = env.step2(2)
        elif state == 'right_collision':
            observation, reward, done = env.step2(3)  

    

def chooseNoRandomAction(RL, observation):
    return RL.choose_action(str(observation),1)
      
def chooseAction (episode, RL, observation):
    if episode < -1:
        return RL.choose_action(str(observation), 0.9 + episode * 0.001)
    else:
        return RL.choose_action(str(observation),1)

def learn (episode, RL, action, reward, observation, observation_):
     if episode < 500:
         RL.learn(str(observation), action, reward, str(observation_), 0.03, 0.9)
     elif episode < 1500 and episode >= 500:
         RL.learn(str(observation), action, reward, str(observation_), 0.3-0.0002*(episode-500), 0.9)
     else:
         RL.learn(str(observation), action, reward, str(observation_), 0.001, 0.9)

def plot(reward):
    plt.style.use('seaborn-deep')
    plt.plot(reward,linewidth= 4.3)
    plt.title('Q Learning Total Reward')
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.show() 
    
def stateChecking(alien_agent, key_agent, action):
    directEnvironment = directNearbyEnvironment(key_agent)

    indirectEnvironment = indirectNearbyEnvironment(key_agent)
    if action == 0 and (alien_agent in [directEnvironment[0], indirectEnvironment[0], indirectEnvironment[1]]):
            return 'upward_collision'
    elif action == 1 and (alien_agent in [directEnvironment[1], indirectEnvironment[2], indirectEnvironment[3]]):
            return 'downward_collision'
    elif action == 2 and (alien_agent in [directEnvironment[3], indirectEnvironment[1], indirectEnvironment[3]]):
            return 'right_collision'
    elif action == 3 and (alien_agent in [directEnvironment[2], indirectEnvironment[0], indirectEnvironment[2]]):
            return 'left_collision'
    else:
        return 'no_collision'

def indirectNearbyEnvironment(coordinate):
    upleft = [coordinate[0]-UNIT, coordinate[1]-UNIT, coordinate[2]-UNIT, coordinate[3]-UNIT]
    upright = [coordinate[0]+UNIT, coordinate[1]-UNIT, coordinate[2]+UNIT, coordinate[3]-UNIT]
    downleft = [coordinate[0]-UNIT, coordinate[1]+UNIT, coordinate[2]-UNIT, coordinate[3]+UNIT]
    downright = [coordinate[0]+UNIT, coordinate[1]+UNIT, coordinate[2]+UNIT, coordinate[3]+UNIT]
    nearby = [upleft, upright, downleft, downright]
    return nearby

def directNearbyEnvironment(coordinate):
    left = [coordinate[0]-UNIT, coordinate[1], coordinate[2]-UNIT, coordinate[3]]
    right = [coordinate[0]+UNIT, coordinate[1], coordinate[2]+UNIT, coordinate[3]]
    up = [coordinate[0], coordinate[1]-UNIT, coordinate[2], coordinate[3]-UNIT]
    down = [coordinate[0], coordinate[1]+UNIT, coordinate[2], coordinate[3]+UNIT]
    nearby = [up, down, left, right]
    return nearby

def humanWalk(humanWalkHelper):
    if humanWalkHelper % 2 == 0:
        if humanWalkHelper/2 < len(HUMANWALK1):
            human = env.humanStep1(HUMANWALK1[int(humanWalkHelper/2)])
        else:
            human = env.humanStep1(4)
    else:
        human = env.humanStep1(4)
    return human
    
if __name__ == "__main__":
    env = Maze()
    RL1 = QLearningTable1(actions=list(range(env.n_actions)),state=CURRENTSTATE)
    RL2 = QLearningTable2(actions=list(range(env.n_actions)),state=CURRENTSTATE)
    backupRL1 = QLearningTable1(actions=list(range(env.n_actions)),state=STATE1)
    backupRL2 = QLearningTable2(actions=list(range(env.n_actions)),state=STATE1)
    ReturnRL1 = ReturnQLearningTable1(actions=list(range(env.n_actions)),state=CURRENTSTATE)
    ReturnRL2 = ReturnQLearningTable2(actions=list(range(env.n_actions)),state=CURRENTSTATE)
    env.after(100, update)
    env.mainloop()
