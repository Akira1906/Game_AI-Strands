import itertools
import copy
import math
import os
import random
import sys
import time
import timeit
import concurrent.futures
from multiprocessing import Process, freeze_support, set_start_method, Pool, cpu_count, Value, Lock, Manager
import threading
import asyncio
from memory_profiler import profile
import tracemalloc
import psutil




# def iterative_deepening_futures(hexes_by_label, curr_board, is_me_player_black, max_time, curr_turn, game_round_number):
#     start_time = time.time()
#     best_move = None
#     best_utility = float('-inf') if curr_turn == 'black' else float('inf')

#     def time_limit_reached():
#         return (time.time() - start_time) >= max_time

#     depth = 1
#     while not time_limit_reached():
#         print(f"Starting search at depth {depth}")
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(min_max_search, hexes_by_label, curr_board, is_me_player_black, depth, float(
#                 '-inf'), float('inf'), curr_turn == 'black', game_round_number)
#             try:
#                 utility, move = future.result(
#                     timeout=max_time - (time.time() - start_time))
#                 print(
#                     f"Depth {depth} completed: utility={utility}, move={move}")
#                 best_utility = utility
#                 best_move = move
#             except concurrent.futures.TimeoutError:
#                 print(f"Timeout reached at depth {depth}")
#                 # Stop the future executor
#                 break  # Time limit reached, break out of the loop
#         depth += 1

#     if best_move is None:
#         # Fallback to random move if no move was found within the time limit
#         available_moves = [coord for label, hex_list in hexes_by_label.items(
#         ) for coord, _ in hex_list if _['owner'] is None]
#         best_move = random.sample(available_moves, 1)
#         print(f"Fallback to random move: {best_move}")
#     print(f"Best move found: {best_move}")
#     return best_utility, best_move


# async def iterative_deepening(hexes_by_label, curr_board, is_me_player_black, max_time, curr_turn, game_round_number):
#     start_time = time.time()
#     best_move = None
#     best_utility = float('-inf')

#     async def search_depth(depth):
#         nonlocal best_move, best_utility
#         print(f"Starting search at depth {depth}")
#         utility, move = min_max_search(hexes_by_label, curr_board, is_me_player_black, depth, float(
#             '-inf'), float('inf'), curr_turn == 'black', game_round_number)
#         print(f"Depth {depth} completed: utility={utility}, move={move}")
#         if (curr_turn == 'black' and utility > best_utility) or (curr_turn == 'white' and utility < best_utility):
#             best_utility = utility
#             best_move = move

#     depth = 1
#     while (time.time() - start_time) < max_time and depth < 3:
#         try:
#             await asyncio.wait_for(search_depth(depth), timeout=max_time - (time.time() - start_time))
#         except asyncio.TimeoutError:
#             print(f"Timeout reached at depth {depth}")
#             break  # Time limit reached, break out of the loop
#         depth += 1

#     if best_move is None:
#         # Fallback to random move if no move was found within the time limit
#         available_moves = [coord for label, hex_list in hexes_by_label.items(
#         ) for coord, _ in hex_list if _['owner'] is None]
#         best_move = random.sample(available_moves, 1)
#         print(f"Fallback to random move: {best_move}")

#     return best_utility, best_move


# # hexes_by_label: available hexes to choose from grouped by label
# # current_board: copy of the game board
# # current_turn : 'black' or 'white'

# def generate_possible_moves_multithread(hexes_by_label, curr_board, player_color, game_round_number):
#     possible_moves = []
#     combinations_cache = {}

#     def process_combination(label, combination):
#         coordinates = [hex_info[0] for hex_info in combination]
#         continuous_neighbors = count_continuous_neighbors(coordinates)
#         if (label == 5 and len(hexes_by_label[label]) > 15 and continuous_neighbors < 3) or \
#            (game_round_number < 5 and continuous_neighbors < label):
#             return

#         board_copy = copy.deepcopy(curr_board)
#         hexes_by_label_copy = copy.deepcopy(hexes_by_label)

#         for hex_info in combination:
#             board_copy[hex_info[0]]['owner'] = player_color
#             board_copy[hex_info[0]]['selected'] = True

#         for hex_info in combination:
#             hexes_by_label_copy[label].remove(hex_info)

#         possible_moves.append((hexes_by_label_copy, board_copy, combination))

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for label in list(hexes_by_label.keys()):
#             hex_list = hexes_by_label[label]
#             label = int(label)

#             if label not in combinations_cache:
#                 if len(hex_list) < label:
#                     combinations_cache[label] = [hex_list]
#                 else:
#                     combinations_cache[label] = list(
#                         itertools.combinations(hex_list, label))

#             for combination in combinations_cache[label]:
#                 futures.append(executor.submit(
#                     process_combination, label, combination))

#         concurrent.futures.wait(futures)

#     return possible_moves


def generate_promising_moves_singlethread(hexes_by_label, curr_board, player_color, game_round_number):
    promising_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label = int(label)

        if len(hex_list) < label:
            move_combinations = [hex_list]
        else:
            move_combinations = gen_combinations(hex_list, label)

        for move in move_combinations:
            if not is_promising_move(len(hexes_by_label[label]), move, game_round_number, label):
                continue

            board_copy = copy.deepcopy(curr_board)
            hexes_by_label_copy = copy.deepcopy(hexes_by_label)

            for hex_info in move:
                board_copy[hex_info[0]]['owner'] = player_color

            for hex_info in move:
                hexes_by_label_copy[label].remove(hex_info)

            promising_moves.append(
                (hexes_by_label_copy, board_copy))
    
    return promising_moves


def generate_promising_moves(hexes_by_label, curr_board, is_player_black, game_round_number):
    player_color = 'black' if is_player_black else 'white'
    return generate_promising_moves_singlethread(hexes_by_label, curr_board, player_color, game_round_number)

def gen_combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def recursive_min_max_search(hexes_by_label, curr_board, is_me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number):
    """
    Performs a min-max search on the game tree to evaluate possible moves.

    Args:
        hexes_by_label (dict): Available hexes to choose from grouped by label.
        curr_board (dict): Copy of the game board.
        is_me_player_black (bool): Indicates if the current player is black.
        remaining_depth (int): Remaining depth of the search.
        max_mode (bool): Indicates if it's the max player's turn.

    Returns:
        Utility of the game position.
    """

    if remaining_depth == 0:
        return evaluate_board_position(curr_board, is_me_player_black, game_round_number), []

    is_curr_player_black = is_me_player_black ^ (not max_mode)
    curr_player_color = 'black' if is_curr_player_black else 'white'
    best_moves = []
    if max_mode:
        best_utility = float('-inf')
    else:
        best_utility = float('inf')

    for label in hexes_by_label.keys():
        hex_list = hexes_by_label[label]
        label = int(label)
        # print("hex_list: " + str(hex_list) + " label: " + str(label))


        if len(hex_list) < label:
            move_combinations = [hex_list]
        else:
            move_combinations = gen_combinations(hex_list, label)

        for move in move_combinations:
            # print("combination: " + str(combination))
            if not is_promising_move(len(hexes_by_label[label]), move, game_round_number, label):
                continue

            for hex_field in move:
                curr_board[hex_field[0]]['owner'] = curr_player_color
            
            hexes_by_label[label] = [hex_field for hex_field in hexes_by_label[label] if hex_field not in move]

            utility, moves = recursive_min_max_search(
                hexes_by_label, curr_board, is_me_player_black, remaining_depth - 1, alpha, beta, not max_mode, game_round_number + 1)

            for hex_field in move:
                curr_board[hex_field[0]]['owner'] = None

            for hex_field in move:
                hexes_by_label[label].append(hex_field)

            if max_mode:
                if utility > best_utility:
                    moves = tuple([hex_field[0] for hex_field in move])

                    best_utility = utility
                    best_moves = moves
                alpha = max(alpha, utility)
            else:
                if utility < best_utility:
                    moves = tuple([hex_field[0] for hex_field in move])

                    best_utility = utility
                    best_moves = moves
                beta = min(beta, utility)
            if beta <= alpha:
                break

    return best_utility, best_moves


def is_promising_move(len_fields_of_label, move, game_round_number, label):
    continuous_neighbors, cluster_dimension = count_continuous_neighbors_and_length(
        [hex_info[0] for hex_info in move])
    # MAX: 34 rounds
    # start phase: 0-10
    # mid phase: 10-20
    # end phase: 20-34
    early_start_phase = 4
    start_phase = 10
    mid_phase = 20
    end_phase = 34

    if game_round_number < early_start_phase:
        if label in [2, 3] and continuous_neighbors < label:
            return False
        elif label == 5 and continuous_neighbors < 4:
            return False

    elif game_round_number < start_phase:
        if label in [2, 3] and continuous_neighbors < label:
            return False
        elif label == 5 and continuous_neighbors < 3:
            return False

    elif game_round_number < mid_phase:
        if label == 2 and len_fields_of_label > 10:
            if continuous_neighbors < 2:
                return False
            
        elif label == 3 and len_fields_of_label > 10:
            if continuous_neighbors < 3:
                return False

        # elif label == 5 and len_fields_of_label > 15:
        #     if cluster_dimension < 4 and continuous_neighbors < 4:
        #         return False

    # if label == 6:
    #     return True
    return True

def start_thread(hexes_by_label, curr_board, is_me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number, lock, memory_debug):
    if (memory_debug): tracemalloc.start()

    with lock:
         local_alpha = alpha.value
    local_beta = float('inf')
    # local_alpha = -float('inf')
    best_move = recursive_min_max_search(hexes_by_label, curr_board, is_me_player_black, remaining_depth, local_alpha, local_beta, max_mode, game_round_number)
    # Stop tracing and get the current, peak and cumulative memory usage
    if (memory_debug):
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        memory_peak = -1
    # print(f"Peak memory usage is {peak / 10**6}MB")
    # update global alpha and beta
    with lock:
        if not max_mode: #inversed because we get max_mode as inversed
            alpha.value = max(alpha.value, best_move[0])
        else:
            beta.value = min(beta.value, best_move[0])
    return best_move, memory_peak



def init_min_max_search(hexes_by_label, curr_board, is_me_player_black, remaining_depth, max_mode, game_round_number):
    memory_debug = False

    if remaining_depth == 0:
        print("error: remaining depth is 0 at init")
    if not max_mode:
        print("error: max mode is false at init")

    # set number of processes
    number_of_processes = cpu_count() - 5
    # print("number of processes: " + str(number_of_processes))
    # generate sub tasks for the processes
    # print all the variables at this point for debugging
    # print("Starting min max search with depth: " + str(remaining_depth) + " and max mode: " + str(max_mode) + " and game round number: " + str(game_round_number))
    # print("hexes by label: " + str(sys.getsizeof(hexes_by_label)))
    # print("current board: " + str(sys.getsizeof(curr_board)))
    # print("is me player black: " + str(is_me_player_black))
    promising_moves = generate_promising_moves(
        hexes_by_label, curr_board, is_me_player_black, game_round_number)
    # import pickle

    # with open(f'promising_moves_{game_round_number}.pickle', 'wb') as f:
    #     pickle.dump(promising_moves, f) 
    # with open(f'promising_moves_8.pickle', 'rb') as f:
    #     promising_moves_8 = pickle.load(f)
    # with open(f'promising_moves_10.pickle', 'rb') as f:
    #     promising_moves_10 = pickle.load(f)

    random.shuffle(promising_moves)
    print(f"Generated {len(promising_moves)} first level branches")
    # if memory_debug:
    #     next_round_moves = generate_promising_moves(hexes_by_label, curr_board, is_me_player_black, game_round_number + 1)
    #     print(f"Generated potential: {str(len(next_round_moves))} second level branches)")
    #     next_round_moves = generate_promising_moves(hexes_by_label, curr_board, is_me_player_black, game_round_number + 2)
    #     print(f"Generated potential: {str(len(next_round_moves))} third level branches")

    # monitoring_thread = threading.Thread(target=monitor_memory, daemon=True)
    # monitoring_thread.start()

    # if game_round_number >= 9:
    #     memory_debug = True
    #     promising_moves = promising_moves[:1]

    # Create a Manager object
    with Manager() as manager:
        # Create Value and Lock objects using the Manager
        alpha = manager.Value('d', -float('inf'))  # 'd' indicates a double precision float
        beta = manager.Value('d', float('inf'))
        lock = manager.Lock()

        with Pool(processes=number_of_processes) as pool:
            results = pool.starmap(start_thread, 
                                   [(hex, board, is_me_player_black, remaining_depth - 1, alpha,
                                      beta, not max_mode, game_round_number + 1, lock, memory_debug)
                                                            for hex, board in promising_moves])             
    moves, memory_usages = zip(*results)
    if memory_debug:
        print(f"Memory usage of all processes: {sum(memory_usages)/ 10**6}MB ")
        print(f"Average memory usage of processes: {(sum(memory_usages)/ len(memory_usages))/ 10**6}MB")
    return max(moves, key=lambda x: x[0])

def monitor_memory(interval=10):
    while True:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory Usage: RSS={memory_info.rss / (1024 * 1024):.2f} MB")
        time.sleep(interval)

def get_neighbors(coord):
    """
    Returns a list of neighboring coordinates for a given coordinate in a hexagonal grid.

    Args:
        coord (tuple): The coordinate for which neighbors are to be found.

    Returns:
        list: A list of neighboring coordinates.
    """
    # Define the six possible relative positions of neighbors in a hexagonal grid
    neighbors = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
    return [(coord[0] + dx, coord[1] + dy) for dx, dy in neighbors]


def count_continuous_neighbors_and_length(coordinates):
    if not coordinates:
        return 0, 0
    coordinates_set = set(coordinates)
    visited = set()

    def flood_fill(coord):
        stack = [coord]
        cluster_size = 0
        cluster = []
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster_size += 1
                cluster.append(current)
                for neighbor in get_neighbors(current):
                    if neighbor in coordinates_set and neighbor not in visited:
                        stack.append(neighbor)
        return cluster, cluster_size

    max_cluster = []
    max_cluster_size = 0
    for coord in coordinates:
        if coord not in visited:
            cluster, cluster_size = flood_fill(coord)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                max_cluster = cluster

    x_coords = [x for x, y in max_cluster]
    y_coords = [y for x, y in max_cluster]
    try:
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        max_cluster_dimension = max(width, height)
    except ValueError:
        print("ValueError")
        max_cluster_dimension = 0
    

    return max_cluster_size, max_cluster_dimension


def count_continuous_neighbors(coordinates):
    coordinates_set = set(coordinates)
    visited = set()

    def flood_fill(coord):
        stack = [coord]
        cluster_size = 0
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster_size += 1
                for neighbor in get_neighbors(current):
                    if neighbor in coordinates_set and neighbor not in visited:
                        stack.append(neighbor)
        return cluster_size

    max_cluster_size = 0
    for coord in coordinates:
        if coord not in visited:
            cluster_size = flood_fill(coord)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size

    return max_cluster_size


def are_neighbors(coord1, coord2):
    # Define the six possible relative positions of neighbors in a hexagonal grid
    neighbors = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]

    # Calculate the relative position between the two coordinates
    relative_position = (coord2[0] - coord1[0], coord2[1] - coord1[1])

    # Check if the relative position is one of the six possible neighbors
    return relative_position in neighbors


def count_hexagons_of(curr_board, player_color):
    owner_count = 0
    for hex_info in curr_board.values():
        owner = hex_info['owner']
        if owner is player_color:
            owner_count += 1
    return owner_count


def evaluate_board_position(curr_board, player_black, game_round_number):
    color = 'black' if player_black else 'white'
    enemy_color = 'white' if player_black else 'black'
    own_connected_area = count_connected_area(curr_board, color)
    enemy_connected_area = count_connected_area(curr_board, enemy_color)
    own_total_area = count_hexagons_of(curr_board, color)
    enemy_total_area = count_hexagons_of(curr_board, enemy_color)
    connected_factor = game_round_number * 0.05

    return (own_connected_area - enemy_connected_area) * connected_factor + own_total_area - enemy_total_area


def count_connected_area(curr_board, player_color):
    # Calculates the largest connected area of hexes of the specified color.

    def dfs(row, col, visited):
        if (row, col) in visited or not (row, col) in curr_board or curr_board[(row, col)]['owner'] != player_color:
            return 0
        visited.add((row, col))
        count = 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            next_row, next_col = row + dr, col + dc
            count += dfs(next_row, next_col, visited)
        return count

    visited = set()
    max_area = 0
    for (row, col), info in curr_board.items():
        if (row, col) not in visited and info['owner'] == player_color:
            area = dfs(row, col, visited)
            if area > max_area:
                max_area = area
    return max_area
