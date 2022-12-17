from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt


cmnds = ['LEFT', 'RIGHT', 'UP', 'DOWN']
dirs = {(0,1),(1,0),(0,-1),(-1,0)}

def strToSchema(s:str):
        return [1 if c=='X' else 0 for c in s.split('\n')[0] ]

def generate_grid(filename):
    f = open(filename,'r')

    schema = [strToSchema(x) for x in f.readlines()] 
    grid = np.array(schema)
    n_zeros = np.count_nonzero(grid==0)
    # print(n_zeros)

    # print('Probability that the drone is in the top left corner:', 1/n_zeros)

    return grid


def transition(p, command, grid):
   
    rows, cols = grid.shape
    
    p_updated = np.zeros_like(p)
    
    for i in range(rows):
        for j in range(cols):
            
            if grid[i, j] == 1:
                continue
      
            if command == "LEFT":
                i_new, j_new = i, j-1
            elif command == "RIGHT":
                i_new, j_new = i, j+1
            elif command == "UP":
                i_new, j_new = i-1, j
            elif command == "DOWN":
                i_new, j_new = i+1, j
            else:
                raise ValueError("Invalid command")
            
            if i_new < 0 or i_new >= rows or j_new < 0 or j_new >= cols or grid[i_new, j_new] == 1:
                p_updated[i, j] += p[i, j]
                
            # the probability is transferred to the new cell
            else:
                p_updated[i_new, j_new] += p[i, j]
    
    return p_updated

def shortest_sequence(grid, initial_prob):
    queue = deque()
    prob_store = {}
    temp_list = [0]*len(cmnds)
    action_list = []
    iter = 0
    queue.append((initial_prob, []))

    while not (len(queue)==0):
        print("[Fringe size : %d] "%(len(queue)),end="\r")
        p, seq = queue.popleft()
        # print("====================================================== FRINGE =====================================================")
        # # for i in queue:
        # #     print(i[1])
        # print(p, seq)
        # print("===================================================================================================================")    
        
        if np.count_nonzero(p==0.0) == grid.shape[0] * grid.shape[1] - 1:
            return seq
        
        for i in range(0, len(cmnds)):
            prob_store[cmnds[i]] = transition(p, cmnds[i], grid)
            temp_list[i] = np.count_nonzero(prob_store[cmnds[i]]==0)
        
        max_val = max(temp_list)
        options = [cmnds[i] for i in range(0,len(temp_list)) if temp_list[i]==max_val]
        # index = random.choice(options)
        for i in range(0,len(options)):
            if len(options) == 1 or (len(seq) and options[i] != seq[-1]) or len(seq) == 0:
                queue.append((prob_store[options[i]], seq + [options[i]]))
        # queue.append((prob_store[index], seq + [cmnds[index]]))

    return None       



filename = 'Thor23-SA74-VERW-Schematic (Classified).txt'
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









