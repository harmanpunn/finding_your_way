from collections import deque
import numpy as np
import random
from heapq import heappush, heappop
import time
from queue import PriorityQueue
import operator
from greedy import greedy


cmnds = ['UP', 'DOWN', 'LEFT', 'RIGHT']
dirs = {(0,1),(1,0),(0,-1),(-1,0)}

def strToSchema(s:str):
        return [1 if c=='X' else 0 for c in s.split('\n')[0] ]

def generate_grid(filename):
    f = open(filename,'r')

    schema = [strToSchema(x) for x in f.readlines()] 
    grid = np.array(schema)
    n_zeros = np.count_nonzero(grid==0)
    # print('Probability that the drone is in the top left corner:', 1/n_zeros)
    return grid


def transition(p, command, grid, cost):
    rows, cols = grid.shape
    p_updated = np.zeros_like(p)
    heuristic = 0 
    
    for i in range(rows):
        for j in range(cols):
            
            if grid[i, j] == 1 or p[i, j] == 0:
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

    # for i in range(rows):
    #     for j in range(cols):
    #         if p_updated[i][j]!=0:
    #             # heuristic += (1/p_updated[i, j])
    #             heuristic += 1/np.min(p_updated[i, j])
    #             # heuristic = (1/np.count_nonzero(p_updated==0.0)) 
    # if np.min(p_updated) != 0: 
    #     heuristic = 1/np.min(p_updated)              
    # print(heuristic, command)
    return p_updated, heuristic  + cost
 

def shortest_sequence(grid, initial_prob):
    bestSeq = None
    for i in range(0,10):
        greedySeq = greedy(grid, initial_prob)
        if bestSeq is None:
            bestSeq = greedySeq
        elif len(greedySeq) < len(bestSeq):
            bestSeq = greedySeq    

    prob_store = {}
    temp_list = [0]*len(cmnds)
    print(len(bestSeq))
    queue = [(0, initial_prob, [])]
    hdict = {}
    newCost = {}
    costDict = {}
    
    costDict[initial_prob.tobytes()] = 0

    def getCost(p):
        return costDict[p.tobytes()]

    def getHeuristic(p):
        rows,cols = p.shape
        h = 0
        for i in range(rows):
            for j in range(cols):
                if p[i][j]!=0:
                    h += (1/p[i, j])
        return h



    while not (len(queue)==0):
        print("[Fringe size : %d] "%(len(queue)),end="\r")
        
        cost, p, seq = min(queue, key = lambda t: t[0])
        queue.remove((cost, p, seq))
        # costDict[p.tobytes()] = cost
        # print('-----------------------------------------------')
        # print(cost, p, seq)
        # print('-----------------------------------------------')

        if not bestSeq is None and len(seq) >= len(bestSeq):
            # print('pruned space')
            continue       

        if np.count_nonzero(p==0.0) == grid.shape[0] * grid.shape[1] - 1:
            print("GOAL STATE")
         
            if len(bestSeq)>len(seq):
                bestSeq =seq
                # print('len(bestSeq)>len(seq):',seq)
                print('len(bestSeq)>len(seq) | seq:',len(seq))
                print('BestSeq update there fore')
            # return seq
            continue
        
        for i in range(0, len(cmnds)):
            prob_store[cmnds[i]],hdict[cmnds[i]] = transition(p, cmnds[i], grid, cost)
            temp_list[i] = np.count_nonzero(prob_store[cmnds[i]]==0)
        
        max_val = max(temp_list)
       
        options = [cmnds[i] for i in range(0,len(temp_list)) if temp_list[i]==max_val]
        for i in range(0,len(options)):
        # if len(options) == 1 or (len(seq) and options[i] != seq[-1]) or len(seq) == 0  :
            newCost = cost +1
            priority = newCost + getHeuristic(prob_store[options[i]])
            if prob_store[options[i]].tobytes() not in costDict or newCost < getCost(prob_store[options[i]]):
                costDict[prob_store[options[i]].tobytes()] =  newCost
                queue.append((priority, prob_store[options[i]], seq + [options[i]]))

    return bestSeq       

def start(schema, commands):
    filename = schema
    grid = generate_grid(filename)
    
    rows, cols = grid.shape
    p = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 1:
                p[i][j] = 1.0 / ((rows * cols) - np.count_nonzero(grid))

    res = shortest_sequence(grid, p)
    print("Best Sequence: ",res)
    print(len(res))  

start('sample5.txt', cmnds)








