
import numpy as np
from queue import PriorityQueue
from greedy import greedy

cmnds = ['LEFT', 'RIGHT', 'UP', 'DOWN']
# cmnds = ['UP', 'RIGHT', 'DOWN', 'LEFT']

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
    print(command)
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

def getTopLeftNonZero(p):
    rows, cols = p.shape
    x = rows
    y = cols
    for i in range(0, rows):
        for j in range(0, cols):
            if p[i][j] != 0:
                y = min(y, j)
    for i in range(0, rows):
        if p[i][y] !=0:
            x = min(x,i)
    return x,y

def getBottomRightNonZero(p):
    rows, cols = p.shape
    x=0
    y=0
    for i in range(0, rows):
        for j in range(0, cols):
            if p[i][j] != 0:
                y = max(y, j)
    for i in range(0, rows):
        if p[i][y] !=0:
            x = max(x,i)
    return x,y         

def manhattan_distance(p1, p2):
    # Unpack the coordinates of the two points
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the Manhattan distance as the sum of the absolute differences of the coordinates
    return abs(x1 - x2) + abs(y1 - y2)

def getMove(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1 and y2 == y1:
        return 'NO MOVE'
    if x2 == x1 and y2 < y1:
        return 'LEFT'
    if x2 == x1 and y2 > y1:
        return 'RIGHT'
    if x2 > x1 and y2 == y1:
        return 'DOWN'
    if x2 < x1 and y2 == y1:
        return 'UP'            
    
def shortest_sequence(grid, lt, rb):
    rows, cols = grid.shape
    queue = PriorityQueue()
    
    queue.put((0,lt))

    costDict = {}
    costDict[lt] = 0

    parent = {}
    parent[lt] = None
    
    while not queue.empty():
        # lt_updated = (0,0)
        pr, curr = queue.get()
        # print('Current Point: ',curr)

        if curr == rb:
            break

        for c in cmnds:
            if c == "LEFT":
                lt_updated = (curr[0], curr[1]-1)
            elif c == "RIGHT":
                lt_updated = (curr[0], curr[1]+1)
            elif c == "UP":
                lt_updated = (curr[0]-1, curr[1])
            elif c == "DOWN":
                lt_updated = (curr[0]+1, curr[1])

            next_cost = costDict[curr] + 1   
            i_new, j_new = lt_updated   
            if (i_new>=0 and i_new<rows and j_new>=0 and j_new<cols and grid[i_new][j_new]!=1) and (lt_updated not in costDict or next_cost < costDict[lt_updated]):
                print('Inside If:', lt_updated, next_cost, c)
                costDict[lt_updated] = next_cost
                parent[lt_updated] = curr
                priority = next_cost +  manhattan_distance(lt_updated, rb) 
                queue.put((priority, lt_updated)) 

    if rb not in parent:
        print('There is no path from lt tp rb')
        return None

    # print(parent)
    move = getMove(parent[rb], rb)
    # print(move)
    return move

def start(schema):
    filename = schema
    grid = generate_grid(filename)
    rows, cols = grid.shape
    print(grid)
    p = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 1:
                p[i][j] = 1.0 / ((rows * cols) - np.count_nonzero(grid))

    l = getTopLeftNonZero(p)
    r = getBottomRightNonZero(p)
    print(p)
    # print(l,r)
    sequence = []
    while l!=r:
        print(l,r)
        move = shortest_sequence(grid,l,r)
        p = transition(p, move, grid)
        sequence.append(move)
        
        l = getTopLeftNonZero(p)
        r = getBottomRightNonZero(p)

        # print(l,r)

    print(sequence)


start('sample3.txt')