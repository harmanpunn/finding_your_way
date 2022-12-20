import numpy as np


cmnds = ['UP', 'LEFT', 'DOWN', 'RIGHT']

# Setting 'X' = 1 ie. blocked and '_' = 0 ie. unblocked
def strToSchema(s:str):
        return [1 if c=='X' else 0 for c in s.split('\n')[0] ]

# Generates a grid from the schema text file
def generate_grid(filename):
    f = open(filename,'r')

    schema = [strToSchema(x) for x in f.readlines()] 
    grid = np.array(schema)
    n_zeros = np.count_nonzero(grid==0)
    return grid

# Transition method to update the beliefs for a given move
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
            
            # Checking if move is valid
            if i_new < 0 or i_new >= rows or j_new < 0 or j_new >= cols or grid[i_new, j_new] == 1:
                p_updated[i, j] += p[i, j]
                
            # the probability is transferred to the new cell
            else:
                p_updated[i_new, j_new] += p[i, j]
    
    return p_updated

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