from collections import deque
import numpy as np
import random
from helper import generate_grid, transition, cmnds
from greedy import greedy

def shortest_sequence(grid, initial_prob):
    queue = deque()
    prob_store = {}
    temp_list = [0]*len(cmnds)
    action_list = []
    iter = 0
    queue.append((initial_prob, []))
    # Getting upper bound using greedy approach
    bestSeqUpdate = greedy(grid, initial_prob)

    while not (len(queue)==0):
        print("[Fringe size : %d] "%(len(queue)),end="\r")
        p, seq = queue.popleft()

        #  We know the the bestSequence will be less than equal to the one returned by greedy approach
        if not bestSeqUpdate is None and len(seq) >= len(bestSeqUpdate):
            continue 
     
        if np.count_nonzero(p==0.0) == grid.shape[0] * grid.shape[1] - 1:
            print("Goal State | Sequence Length: ", len(seq))
            if len(bestSeqUpdate)>len(seq):
                bestSeqUpdate = seq
            continue
        
        for i in range(0, len(cmnds)):
            prob_store[cmnds[i]] = transition(p, cmnds[i], grid)
            temp_list[i] = np.count_nonzero(prob_store[cmnds[i]]==0)
        
        max_val = max(temp_list)
        options = [cmnds[i] for i in range(0,len(temp_list)) if temp_list[i]==max_val]
        for i in range(0,len(options)):
            if len(options) == 1 or (len(seq) and options[i] != seq[-1]) or len(seq) == 0:
                queue.append((prob_store[options[i]], seq + [options[i]]))

    return bestSeqUpdate       


def start(schema):
    filename = schema
    grid = generate_grid(filename)
    print(grid)

    rows, cols = grid.shape
    p = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 1:
                p[i][j] = 1.0 / ((rows * cols) - np.count_nonzero(grid))

    res = shortest_sequence(grid, p)
    print("Best Sequence: ",res)
    print(len(res))    


start('sample3.txt')