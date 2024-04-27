import numpy as np
import pandas
import heapq
import copy as cp
from collections import deque

Pos = {
    "NORTH": 0,
    "SOUTH": 2,
    "WEST": 3,
    "EAST": 1
}

DATA_QUANTITY = 48 + 1 #num 0 (unused), 1, ... , 48
DEFPOSTOCWPOS = [Pos["NORTH"], Pos["SOUTH"], Pos["WEST"], Pos["EAST"]] # N, E, S, W = 0, 1, 2, 3
DIR = ['f', 'r', 'b', 'l']
STARTING_POINT = 2

WIDTH = 6
LENGTH = 8

adj_list = np.full((DATA_QUANTITY, 4), -1)

nodes_weight = np.zeros(DATA_QUANTITY)
nodes_visited = np.full(DATA_QUANTITY, False)
nodes_prev = np.full(DATA_QUANTITY, -1)
nodes_prevpos = np.full(DATA_QUANTITY, -1)

valued_nodes_index = np.full(shape=DATA_QUANTITY, dtype=int, fill_value=0)
valued_nodes_weight = np.full(shape=DATA_QUANTITY, dtype=int, fill_value=0)
valued_nodes_visited = np.full(DATA_QUANTITY, False)
valued_nodes_num = 0

pivot_nodes_list = None # a np.array full of tuples

next_command = np.full(shape=DATA_QUANTITY, dtype=object, fill_value="-1")
waiting_list = deque()

starting_point = 0

def downloadMap(filepath: str):
    raw_data = pandas.read_csv(filepath).values
    raw_data[np.isnan(raw_data)] = 0

    adj_list[1: DATA_QUANTITY, 0:] = raw_data[:DATA_QUANTITY - 1, 1:5]

    global valued_nodes_num
    cur_index = 0
    for i in range(len(raw_data)):
        if (np.count_nonzero(raw_data[i][1:5]) == 1):
            valued_nodes_index[cur_index] = i + 1
            valued_nodes_weight[cur_index] = 0#To Be Done 
            valued_nodes_num += 1
            cur_index += 1

    starting_point = STARTING_POINT


def changeStart(start: int):
    starting_point = start

def printArray(arr: np.array):
    for t in range(len(arr)):
        for sth in arr[t]:
            print(sth, end=" ");
        print("\n")

def BFS(start: int, end: int) -> str:
    trav_list = np.array([], dtype="int8")
    travpos_list = np.array([], dtype="int8")
    isSearching = True
    #now_pos = 'x'
    start_pos = 0

    nodes_visited.fill(0)
    nodes_prevpos[start] = -1

    for i in range(4):
        if (adj_list[start][i]):
            start_pos = DEFPOSTOCWPOS[i]
            break

    waiting_list.clear()
    waiting_list.appendleft(start)
    nodes_visited[start] = 1

    while isSearching:
        for node in waiting_list.copy():
            waiting_list.pop();
            for i in range(4):
                next_node = adj_list[node][i]
                if (nodes_visited[next_node] or next_node == -1): continue
                waiting_list.appendleft(next_node)
                nodes_visited[next_node] = True
                nodes_prev[next_node] = node
                nodes_prevpos[next_node] = DEFPOSTOCWPOS[i]
                if (next_node == end): isSearching = False
        if (len(waiting_list) == 0 and isSearching): return np.array([-1])

    prev_node = -1
    prev_nodepos = -1
    while prev_node != start:
        prev_node = end if prev_node == -1 else nodes_prev[prev_node]
        prev_nodepos = start_pos if nodes_prevpos[prev_node] == -1 else nodes_prevpos[prev_node]

        trav_list = np.insert(trav_list, 0, prev_node, axis=0)
        travpos_list = np.insert(travpos_list, 0, prev_nodepos, axis=0)

    travdir_str = posToDir(travpos_list)
    return travdir_str

def posToDir(pos_list: np.ndarray):
    pos_len = len(pos_list)
    dir_num_list = (pos_list[1:] - pos_list[:-1]) % 4
    dir_list = np.array([])
    for dir_num in dir_num_list:
        dir_list = np.append(dir_list, DIR[dir_num])
    return "".join(map(str, dir_list))

def makePathCsv(filepath = "./maze (2).csv"):
    downloadMap(filepath)
    path_table = np.ndarray(shape=[valued_nodes_num + 1, valued_nodes_num + 1], dtype=object)
    path_dist_table = np.ndarray(shape=[valued_nodes_num + 1, valued_nodes_num + 1], dtype=np.uint16)
    path_dist_table.fill(0)
    
    for row in range(1 ,valued_nodes_num + 1):
        path_table[row][0] = valued_nodes_index[row - 1]
        path_dist_table[row][0] = valued_nodes_index[row - 1]
        for col in range(1, valued_nodes_num + 1):
            if (row == col): 
                path_table[row][col] = "-1"
                path_dist_table[row][col] = 0
            else: 
                shortest_path = BFS(valued_nodes_index[row - 1], valued_nodes_index[col - 1]) 
                path_table[row][col] = shortest_path
                path_dist_table[row][col] = len(shortest_path)
    path_data = pandas.DataFrame(path_table.tolist())
    path_dist_data = pandas.DataFrame(path_dist_table.tolist())
    path_data.to_csv(path_or_buf="data//paths.csv", index=False)
    path_dist_data.to_csv(path_or_buf="data//paths_dist.csv", index=False)

def downloadCsv() -> tuple[np.array, np.array]:
    path_data = pandas.read_csv("data//paths.csv").values
    path_dist_data = pandas.read_csv("data//paths_dist.csv").values

    return (path_data, path_dist_data)

def path_init(path_data: np.ndarray, path_dist_data: np.ndarray):
    global valued_nodes_num, valued_nodes_index, valued_nodes_weight
    valued_nodes_index = path_data[:, 0:1].ravel()[1:]
    valued_nodes_index.astype(np.uint8)
    valued_nodes_num = len(valued_nodes_index)
    start_w = (STARTING_POINT - 1) % WIDTH
    start_l = (STARTING_POINT - 1) // WIDTH
    for index in range(len(valued_nodes_index)):
        cur_w = (valued_nodes_index[index] - 1) % WIDTH
        cur_l = (valued_nodes_index[index] - 1) // WIDTH
        valued_nodes_weight[index] = abs(cur_w + cur_l - start_w - start_l)
    #print(valued_nodes_num)
    #print(valued_nodes_weight)  

    starting_point = STARTING_POINT 

def strats(start: int) -> list[str]:
    path_data, path_dist_data = downloadCsv()
    path_init(path_data, path_dist_data)
    path_data = path_data[1:, 1:]
    path_dist_data = path_dist_data[1:, 1:] 

    valued_node_duplicate_list = np.transpose(
                                    np.reshape(
                                        np.repeat(np.arange(valued_nodes_num), valued_nodes_num)
                                    , [valued_nodes_num, valued_nodes_num])
                                 )
    #print(path_dist_data)
    adj_node_and_length_list = np.dstack([valued_node_duplicate_list, path_dist_data])
    for i in range(valued_nodes_num):
        adj_node_and_length_list[i] = adj_node_and_length_list[i][adj_node_and_length_list[i][:, 1].argsort()]
    adj_node_and_length_list = adj_node_and_length_list[:,  1:]
    
    # for i in range(valued_nodes_num):
    #     for nnode in adj_node_and_length_list[i]:
    #         print(nnode)
    # return

    validate_paths = []
    toSearch = 10
    current_node = start
    path_heap = []
    heapq.heappush(path_heap, (0, 0, [start])) #weight, actual_path_length, list
    t = 0
    while toSearch:
        curpath_weight, curpath_length, curpath= heapq.heappop(path_heap)
        #print(path_heap)
        curnode = curpath[-1]
        hasPathLE3 = False
        continuePushing = 6
        if (len(curpath) >= 4): continuePushing = 4
        if (len(curpath) >= 8): continuePushing = 3 
        #if (len(curpath) >= 11): continuePushing = 3
        for next_node in adj_node_and_length_list[curnode]:
            if next_node[0] in curpath: continue
            #print(adj_node_and_length_list[curnode], next_node)
            #next_node: (path_num: int, length: int)
            arr_to_push = cp.deepcopy(curpath)
            arr_to_push.append(next_node[0])
            if (next_node[1] <= 3):
                hasPathLE3 = True
            if (next_node[1] > 5 and hasPathLE3):
                break
            
            buffer_const =  0.8
            if (len(arr_to_push) >= valued_nodes_num * 2 / 3): 
                buffer_const = 0.1
            if (len(arr_to_push) < valued_nodes_num / 3): 
                buffer_const = 0
            if (len(arr_to_push) == valued_nodes_num):
                validate_paths.append((arr_to_push, curpath_length + next_node[1], curpath_weight + next_node[1] - buffer_const * (valued_nodes_weight[next_node[0]] + valued_nodes_weight[curnode])))
                toSearch -= 1 if toSearch >= 1 else 0
            heapq.heappush(path_heap, (curpath_weight + next_node[1] - buffer_const * (valued_nodes_weight[next_node[0]] + valued_nodes_weight[curnode]), curpath_length + next_node[1], arr_to_push))

            if (t % 20000 == 0): 
                #print(valued_nodes_weight)
                #print(valued_nodes_weight[next_node[0]], valued_nodes_index[curnode])
                #print(t, heapq.nsmallest(4, path_heap))
                #print(path_heap)
                #print(t)
                pass
            t += 1

            #print(next_node)
            continuePushing -= 1
            last_length = next_node[1]
            if (not continuePushing): break
    
    #print(validate_paths, sep='/n')
    validate_paths = validate_paths[0]
    opus_magnum = []
    former_node = validate_paths[0][0]
    isFirstPath = True
    isSecondPath = False
    for node in validate_paths[0][1:]:
        #print()
        the_path = path_data[former_node, node][1:]
        former_node = node
        if not isFirstPath:
            the_path = "b" + the_path

        if not isSecondPath:
            opus_magnum.append(the_path)
        else:
            opus_magnum[0] += the_path

        if isSecondPath:
            isSecondPath = False
        elif isFirstPath:
            isFirstPath = False
            isSecondPath = True

    opus_magnum[-1] += "s"
    opus_magnum_int = []
    for command in opus_magnum:
        opus_magnum_int.append([])
        opus_magnum_comm = opus_magnum_int[-1]
        for sing_comm in command:
            opus_magnum_comm.append(int(ord(sing_comm)))
        #print(opus_magnum_int)
    makeOpusMagnumCsv(opus_magnum)
    opus_magnum_int_dframe = pandas.DataFrame(opus_magnum_int)
    #print(opus_magnum_int)
    opus_magnum_int_dframe.to_csv(path_or_buf="data//the_path.csv", index=False)
    return opus_magnum

def makeOpusMagnumCsv(opus_magnum: list):
    opus_magnum_dframe = pandas.DataFrame(opus_magnum)
    opus_magnum_dframe.to_csv(path_or_buf="data//the_path_chr.csv", index=False)

def calcStrat(filepath = "./maze (2).csv", start: int = 0) -> list[str]:
    makePathCsv(filepath)
    return strats(start)







'''if __name__ == "__main__":
    t = 1
    while True:
        sths = input().split(',')
        if sths[0] == "-1":
            break
        for i in range(4):
            if (sths[i + 1] != ''):
                adj_list[t][i] = int(sths[i + 1])
        t += 1
    alpha, omega = tuple([int(x) for x in input(f"start and end:").split(',')])
    print(*BFS(alpha, omega), sep='')'''

if __name__ == "__main__":
    start = STARTING_POINT
    calcStrat("big_maze.csv", start)
    ans = strats(STARTING_POINT)
    print("".join(ans))
    ans[0] = "f" + ans[0]
    ans[-1] = ans[-1][:-1]
    print("w/ f w/o s: ","".join(ans), sep="")

    #downloadMap("./medium_maze.csv")
    #path_data, path_dist_data = downloadCsv()
    #path_data = path_data[1:, 1:]
    #path_dist_data = path_dist_data[1:, 1:] 
    
    #makePathCsv();
    