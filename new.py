from collections import deque
import numpy as np
import random
from heapq import heappush, heappop
import time
from queue import PriorityQueue
import operator
# import matplotlib.pyplot as plt
from greedy import greedy


cmnds = ['LEFT', 'RIGHT', 'UP', 'DOWN']
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


    return p_updated
 

def shortest_sequence(grid, initial_prob):
    bestSeq = None
    for i in range(0,1):
        greedySeq = greedy(grid, initial_prob)
        if bestSeq is None:
            bestSeq = greedySeq
        elif len(greedySeq) < len(bestSeq):
            bestSeq = greedySeq    

    prob_store = {}
    temp_list = [0]*len(cmnds)
    print(len(bestSeq))
    # queue = deque()
    visited = set()
    queue = [(0, initial_prob, [])]
    visited.add(initial_prob.tobytes())
    hdict = {}
    newCost = {}
    costDict = {}
    
    costDict[initial_prob.tobytes()] = 0

    def getCost(p):
        return costDict[p.tobytes()]

    def getHeuristicOld(p):
        rows,cols = p.shape
        h = 0
        for i in range(rows):
            for j in range(cols):
                if p[i][j]!=0:
                    h += (1/p[i, j])
        return h

    def getHeuristicTest(p):
        return 1/np.min(p[p!=0])
        # if np.min(p) != 0: 
        #     return 1/np.min(p)
        # return 0      

    def getHeuristic(p:np.ndarray):
        maxVal = np.max(p)
        listKeys = []
        for i in range(0,p.shape[0]):
            for j in range(0,p.shape[1]):
                if p[i][j]==maxVal:
                    listKeys.append((i,j))
        if len(listKeys)==1:
            secondMax = np.max(p[p!=maxVal])
            listKeysSecond = []
            for i in range(0,p.shape[0]):
                for j in range(0,p.shape[1]):
                    if p[i][j]==secondMax:
                        listKeysSecond.append((i,j))
            h = float("inf")
            for i in range(0,len(listKeysSecond)):
                # h = min(h, abs(listKeys[0][0]-listKeysSecond[i][0])+abs(listKeys[0][1]-listKeysSecond[i][1]))      
                h = min(h, max(listKeys[0][0],listKeysSecond[i][0])+max(listKeys[0][1],listKeysSecond[i][1]))  
            return h
        h = float("inf")
        for i in range(0,len(listKeys)):
            for j in range(0,len(listKeys)):
                if i!=j:
                    # h = min(h, abs(listKeys[i][0]-listKeys[j][0])+abs(listKeys[i][1]-listKeys[j][1]))     
                    h = min(h, max(listKeys[i][0],listKeys[j][0])+max(listKeys[i][1],listKeys[j][1]))      
        return h

    prune_count=0
    seq_length_list = []
    
    while not (len(queue)==0):
        # print("[Fringe size : %d]"%(len(queue)), end="\r")
        # cost, p, seq = queue.popleft()

        # cost, p, seq = min(queue, key = lambda t: t[0])
        # index = queue.index(max(queue, key = lambda t: len(t[2])))
        index = queue.index(min(queue, key = lambda t: t[0]))

        cost, p, seq = queue[index]
        queue.pop(index)

        print("[Fringe size : %d] [Pruned size : %d] [Current Seq size : %d] [Heuristic : %d]"%((len(queue)), prune_count, len(seq) , getHeuristic(p)), end="\r")
        # print('-----------------------------------------------')
        # print(cost, p, seq)
        # print('-----------------------------------------------')

        # We know the the bestSequence will be less than equal to the one returned by greedy approach
        if not bestSeq is None and len(seq) >= len(bestSeq):
            prune_count+=1
            continue       

        if np.count_nonzero(p==0.0) == grid.shape[0] * grid.shape[1] - 1:
            print("GOAL STATE")
            # plt.imshow(p)
            # plt.show()
            if len(bestSeq)>len(seq):
                bestSeq = seq
                print('len(bestSeq)>len(seq) | seq:',len(seq))
                print('BestSeq update there fore')
            continue
        
        for i in range(0, len(cmnds)):
            prob_store[cmnds[i]] = transition(p, cmnds[i], grid, cost)
            temp_list[i] = np.count_nonzero(prob_store[cmnds[i]]==0)
        
        max_val = max(temp_list)
       
        # options = [cmnds[i] for i in range(0,len(temp_list)) if temp_list[i]==max_val]
        options = cmnds
        for i in range(0,len(options)):
            # if len(options) == 1 or (len(seq) and options[i] != seq[-1]) or len(seq) == 0  :
            newCost = cost + 1
            priority = newCost  +  getHeuristicTest(prob_store[options[i]])
            if (prob_store[options[i]].tobytes() not in costDict or newCost < getCost(prob_store[options[i]])) and seq + [options[i]] not in seq_length_list:
                costDict[prob_store[options[i]].tobytes()] =  newCost
                seq_length_list.append(seq + [options[i]])
                queue.append((priority , prob_store[options[i]], seq + [options[i]]))
                visited.add(prob_store[options[i]].tobytes())
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

start('sample3.txt', cmnds)








