from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

class Agent:
  def __init__(self, direction, ioi, start):
    self.direction = direction 
    self.ioi = ioi
    self.start = start
    self.predictions = []  # beat predictions
    self.error = []  # error of beats from closest onset
    
# Agent hypothesises beat path from start to finish
def agent_forward_pass(ioi, i, onsets, max_time):
    a = Agent( 'forward', ioi, onsets[i])
            
    state = a.start    
    while state < max_time:

        # Compute error from state to closest onset
        distances = np.array([abs(state - x) for x in onsets])
        error = min(distances)
        
        a.error.append(error)
        a.predictions.append(state)
        
        # jump by the hypothesised ioi
        state += ioi
        
    return a

# Agent hypothesises beat path from end to beginning
def agent_backward_pass(ioi, i, onsets):
    a = Agent('backward', ioi, onsets[-i])
            
    state = a.start    
    while state > 0:

        # Compute error from state to closest onset
        distances = np.array([abs(state - x) for x in onsets])
        error = min(distances)
        
        a.error.append(error)
        a.predictions.append(state)
        
        # jump backwards by the hypothesised ioi
        state -= ioi
        
    return a
    

# Multiple agents hypothesise beat paths, to later be evaluated
def agent_voting(max_time, onsets, ioi_common, bidirectional=True):
    
    # Number of agents proportionate to number of clusters common ioi and starting onsets
    # Loops through common iois, and all possible start points
    agents = []
    for ioi in ioi_common:
        for i in range(len(onsets)//2): # takes first/last half of onsets as potential starts 
            
            # Create, execute and store new agents
            agents.append(agent_forward_pass(ioi, i, onsets, max_time))
            if bidirectional: agents.append(agent_backward_pass(ioi, i, onsets))
                
    return agents

def best_agent(agents):
    beat_scores = Counter()
    
    # Compute scores per beat based on popularity and onset error
    # disencourages error (added +0.001 in case its zero)
    for a in agents:
        for i, p in enumerate(a.predictions):
            beat_scores[round(p, 3)] += 1 / (a.error[i] + 0.001)
        
    # Compute agent score based on final beat event scores
    agent_scores = []
    for a in agents:
        agent_score = 0
        
        for i, p in enumerate(a.predictions):
            agent_score += beat_scores[round(p, 3)]
            
        agent_scores.append(agent_score)
        
    best_agent = agents[np.argmax(agent_scores)]  
    
    return sorted(best_agent.predictions)