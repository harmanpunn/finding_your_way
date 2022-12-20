import numpy as np
from queue import PriorityQueue
from greedy import greedy
from helper import generate_grid, transition, getMove, cmnds

# Get left-top nonzero point
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


# Get right-bottom nonzero point
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
                # print('Inside If:', lt_updated, next_cost, c)
                costDict[lt_updated] = next_cost
                parent[lt_updated] = (curr, c)
                priority = next_cost +  manhattan_distance(lt_updated, rb) 
                queue.put((priority, lt_updated)) 

    if rb not in parent:
        print('There is no path from lt tp rb')
        return None

    pred_rb = parent[rb]
    path = []
    while not curr is None:
        path.append(pred_rb[1])
        pred_rb = parent[pred_rb[0]]
   
    path.reverse()
    return path[0]

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
    sequence = []
    while l!=r:
        move = shortest_sequence(grid,l,r)
        p = transition(p, move, grid)
        sequence.append(move)
        l = getTopLeftNonZero(p)
        r = getBottomRightNonZero(p)
    print(p)
    print(sequence, len(sequence))


start('reactor.txt')