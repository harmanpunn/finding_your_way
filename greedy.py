
import numpy as np
import random
from helper import generate_grid, transition, cmnds

@staticmethod
def greedy(grid, p):
    temp_list = [0]*len(cmnds)
    action_list = []
    while True:
        prob_store = {}
        for i in range(0, len(cmnds)):
            prob_store[i] = transition(p, cmnds[i], grid)
            temp_list[i] = np.count_nonzero(prob_store[i]==0)
        max_value = max(temp_list)
        options = [i for i in range(0,len(temp_list)) if temp_list[i]==max_value]
        
        index = random.choice(options)
        # print('Action Took:', cmnds[index])
        action_list.append(cmnds[index])
        p = prob_store[index]

        if max_value == grid.shape[0] * grid.shape[1] - 1:
            break
    return action_list


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

    res = greedy(grid, p)
    print("Best Sequence: ",res)
    print(len(res))    

start('reactor.txt')