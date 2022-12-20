import numpy as np
from queue import PriorityQueue
from greedy import greedy
from helper import generate_grid, transition, cmnds


def shortest_sequence(grid, initial_prob):
    bestSeq = None
    '''
    for i in range(0,1):
        greedySeq = greedy(grid, initial_prob)
        if bestSeq is None:
            bestSeq = greedySeq
        elif len(greedySeq) < len(bestSeq):
            bestSeq = greedySeq    
    '''
    prob_store = {}
    
    queue = PriorityQueue()
    queue.put((0, initial_prob.tobytes()))

    costDict = {}
    sequence = {}
    
    costDict[initial_prob.tobytes()] = 0
    sequence[initial_prob.tobytes()] = []

    def getCost(p):
        # return 1 / np.count_nonzero(p==0.0)
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
    
    def getHeuristicv1(p: np.ndarray):
        minY, maxY= p.shape[1],0
        minX, maxX= p.shape[0],0
        for i in range(0,p.shape[0]):
            for j in range(0,p.shape[1]):
                if p[i][j]!=0:
                    minX, maxX =  min(minX,i),max(maxX,i)
                    minY, maxY =  min(minY,j),max(maxY,j)
        return abs(minX-maxX)+abs(minY-maxY)

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
    min_h_so_far = 99999999
    while not queue.empty():
        prio, element = queue.get()
        p = np.frombuffer(element,dtype=initial_prob.dtype).reshape(initial_prob.shape)
        cost = costDict[p.tobytes()]
        seq = sequence[p.tobytes()]

        if not bestSeq is None and prio >= len(bestSeq):
            prune_count+=1
            continue 

        if not bestSeq is None and len(seq) >= len(bestSeq):
            prune_count+=1
            continue 

        print("[Fringe size : %d] [Pruned size : %d] [Current Seq size : %d] [Heuristic : %d]"%(queue.qsize(), prune_count, len(seq) , getHeuristicv1(p)), end="\r")        

        if np.count_nonzero(p==0.0) == grid.shape[0] * grid.shape[1] - 1:
            print("Goal State | Sequence Length: ", len(seq))
            if bestSeq is None or len(len(bestSeq)>len(seq)):
                bestSeq = seq
            continue

        # Belief updates for each of the four possible moves   
        for i in range(0, len(cmnds)):
            prob_store[cmnds[i]] = transition(p, cmnds[i], grid)
        
        options = cmnds

        for i in range(0,len(options)):
            # Updating cost and priority
            newCost = cost + 1
            priority =  newCost + getHeuristicv1(prob_store[options[i]]) 
            if (prob_store[options[i]].tobytes() not in costDict or newCost < getCost(prob_store[options[i]])):
                costDict[prob_store[options[i]].tobytes()] =  newCost
                sequence[prob_store[options[i]].tobytes()] = seq + [options[i]]
                queue.put((priority, prob_store[options[i]].tobytes()))

    return bestSeq       

def start(schema):
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

start('sample5.txt')








