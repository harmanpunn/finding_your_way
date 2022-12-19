
from collections import deque
import numpy as np
import random
from heapq import heappush, heappop
import time

cmnds = ['UP', 'LEFT', 'DOWN', 'RIGHT']

def strToSchema(s:str):
        return [1 if c=='X' else 0 for c in s.split('\n')[0] ]

def generate_grid(filename):
    f = open(filename,'r')

    schema = [strToSchema(x) for x in f.readlines()] 
    grid = np.array(schema)
    n_zeros = np.count_nonzero(grid==0)
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


# def find_drone(commands, grid):
  
#     rows, cols = grid.shape
    
#     # Initialize the probability distribution with uniform probabilities over the unblocked cells
#     p = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             if grid[i][j] != 1:
#                 p[i][j] = 1.0 / ((rows * cols) - np.count_nonzero(grid))
    
#     for command in commands:
#         p = transition(p, command, grid)
        
#         for i in range(p.shape[0]):
#             for j in range(p.shape[1]):
#                 if p[i][j] == 1:
#                     return i,j
       

#     # If the location of the drone has not been determined, return None
#     return None
       
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
        


grid = generate_grid('sample5.txt')
rows, cols = grid.shape        
cmnds = ['LEFT', 'RIGHT', 'UP', 'DOWN']
sequence = []
p = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        if grid[i][j] != 1:
            p[i][j] = 1.0 / ((rows * cols) - np.count_nonzero(grid))

         
# p = [[1.0 / ((rows * cols) - np.count_nonzero(grid)) for _ in range(rows)] for _ in range(cols)]            
# while True:            
#     res = greedy(grid, p)
#     if len(res) == 13:
#         print(res)
#         print(len(res))
#         break
    
      