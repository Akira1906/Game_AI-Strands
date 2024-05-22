import itertools
import copy
import math
import os
import random
import sys
import time
import timeit
import concurrent.futures
from multiprocessing import Process, freeze_support, set_start_method, Pool
import threading
import asyncio


def iterative_deepening_futures(hexes_by_label, curr_board, me_player_black, max_time, curr_turn, game_round_number):
    start_time = time.time()
    best_move = None
    best_utility = float('-inf') if curr_turn == 'black' else float('inf')

    def time_limit_reached():
        return (time.time() - start_time) >= max_time

    depth = 1
    while not time_limit_reached():
        print(f"Starting search at depth {depth}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(min_max_search, hexes_by_label, curr_board, me_player_black, depth, float(
                '-inf'), float('inf'), curr_turn == 'black', game_round_number)
            try:
                utility, move = future.result(
                    timeout=max_time - (time.time() - start_time))
                print(
                    f"Depth {depth} completed: utility={utility}, move={move}")
                best_utility = utility
                best_move = move
            except concurrent.futures.TimeoutError:
                print(f"Timeout reached at depth {depth}")
                # Stop the future executor
                break  # Time limit reached, break out of the loop
        depth += 1

    if best_move is None:
        # Fallback to random move if no move was found within the time limit
        available_moves = [coord for label, hex_list in hexes_by_label.items(
        ) for coord, _ in hex_list if _['owner'] is None]
        best_move = random.sample(available_moves, 1)
        print(f"Fallback to random move: {best_move}")
    print(f"Best move found: {best_move}")
    return best_utility, best_move


async def iterative_deepening(hexes_by_label, curr_board, me_player_black, max_time, curr_turn, game_round_number):
    start_time = time.time()
    best_move = None
    best_utility = float('-inf')

    async def search_depth(depth):
        nonlocal best_move, best_utility
        print(f"Starting search at depth {depth}")
        utility, move = min_max_search(hexes_by_label, curr_board, me_player_black, depth, float(
            '-inf'), float('inf'), curr_turn == 'black', game_round_number)
        print(f"Depth {depth} completed: utility={utility}, move={move}")
        if (curr_turn == 'black' and utility > best_utility) or (curr_turn == 'white' and utility < best_utility):
            best_utility = utility
            best_move = move

    depth = 1
    while (time.time() - start_time) < max_time and depth < 3:
        try:
            await asyncio.wait_for(search_depth(depth), timeout=max_time - (time.time() - start_time))
        except asyncio.TimeoutError:
            print(f"Timeout reached at depth {depth}")
            break  # Time limit reached, break out of the loop
        depth += 1

    if best_move is None:
        # Fallback to random move if no move was found within the time limit
        available_moves = [coord for label, hex_list in hexes_by_label.items(
        ) for coord, _ in hex_list if _['owner'] is None]
        best_move = random.sample(available_moves, 1)
        print(f"Fallback to random move: {best_move}")

    return best_utility, best_move


# hexes_by_label: available hexes to choose from grouped by label
# current_board: copy of the game board
# current_turn : 'black' or 'white'

def generate_possible_moves_multithread(hexes_by_label, curr_board, player_color, game_round_number):
    possible_moves = []
    combinations_cache = {}

    def process_combination(label, combination):
        coordinates = [hex_info[0] for hex_info in combination]
        continuous_neighbors = count_continuous_neighbors(coordinates)
        if (label == 5 and len(hexes_by_label[label]) > 15 and continuous_neighbors < 3) or \
           (game_round_number < 5 and continuous_neighbors < label):
            return

        board_copy = copy.deepcopy(curr_board)
        hexes_by_label_copy = copy.deepcopy(hexes_by_label)

        for hex_info in combination:
            board_copy[hex_info[0]]['owner'] = player_color
            board_copy[hex_info[0]]['selected'] = True

        for hex_info in combination:
            hexes_by_label_copy[label].remove(hex_info)

        possible_moves.append((hexes_by_label_copy, board_copy, combination))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for label in list(hexes_by_label.keys()):
            hex_list = hexes_by_label[label]
            label_int = int(label)

            if label not in combinations_cache:
                if len(hex_list) < label_int:
                    combinations_cache[label] = [hex_list]
                else:
                    combinations_cache[label] = list(
                        itertools.combinations(hex_list, label_int))

            for combination in combinations_cache[label]:
                futures.append(executor.submit(
                    process_combination, label_int, combination))

        concurrent.futures.wait(futures)

    return possible_moves


def generate_possible_moves_singlethread(hexes_by_label, curr_board, player_color, game_round_number):
    possible_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label_int = int(label)

        if len(hex_list) < label_int:
            combinations = [hex_list]
        else:
            combinations = itertools.combinations(hex_list, label_int)

        for combination in combinations:
            continuous_neighbors = count_continuous_neighbors(
                [hex_info[0] for hex_info in combination])
            # continuous_neighbors = random.randint(0, 6)
            if label_int == 5 and len(hexes_by_label[label]) > 15:
                if continuous_neighbors < 3:
                    continue

            if game_round_number < 5:
                if continuous_neighbors < label_int:
                    continue

            board_copy = copy.deepcopy(curr_board)
            hexes_by_label_copy = copy.deepcopy(hexes_by_label)

            for hex_info in combination:
                board_copy[hex_info[0]]['owner'] = player_color
                board_copy[hex_info[0]]['selected'] = True

            for hex_info in combination:
                hexes_by_label_copy[label].remove(hex_info)

            possible_moves.append(
                (hexes_by_label_copy, board_copy, combination))
    return possible_moves


def generate_possible_moves(hexes_by_label, curr_board, player_color, game_round_number):
    return generate_possible_moves_singlethread(hexes_by_label, curr_board, player_color, game_round_number)


def recursive_min_max_search(hexes_by_label, curr_board, me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number):
    """
    Performs a min-max search on the game tree to evaluate possible moves.

    Args:
        hexes_by_label (dict): Available hexes to choose from grouped by label.
        curr_board (dict): Copy of the game board.
        me_player_black (bool): Indicates if the current player is black.
        remaining_depth (int): Remaining depth of the search.
        max_mode (bool): Indicates if it's the max player's turn.

    Returns:
        Utility of the game position.
    """

    curr_player_black = me_player_black ^ (not max_mode)
    player_color = 'black' if curr_player_black else 'white'

    if remaining_depth == 0:
        return evaluate_board_position(curr_board, me_player_black, game_round_number), []

    best_moves = []
    if max_mode:
        best_utility = float('-inf')
    else:
        best_utility = float('inf')
    # possible_moves = []
    # print("max: generate possible moves...")
    # before = time.time()
    # # generate the possible moves (leaves)
    # possible_moves = generate_possible_moves(hexes_by_label, curr_board, player_color, game_round_number)
    # print('max: generated possible moves: ' + str(len(possible_moves)))
    # random.shuffle(possible_moves)
    # after = time.time()
    # print('max: generate possible moves time: ' + str(after - before))

    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label_int = int(label)

        if len(hex_list) < label_int:
            combinations = [hex_list]
        else:
            combinations = itertools.combinations(hex_list, label_int)

        for combination in combinations:
            continuous_neighbors = count_continuous_neighbors(
                [hex_info[0] for hex_info in combination])
            # # continuous_neighbors = random.randint(0, 6)
            if label_int == 5 and len(hexes_by_label[label]) > 15:
                if continuous_neighbors < 3:
                    continue

            if label_int == 2 and game_round_number < 10:
                if continuous_neighbors < 2:
                    continue

            if game_round_number < 30:
                if continuous_neighbors < label_int-1:
                    continue

            # board_copy = copy.deepcopy(curr_board)
            # hexes_by_label_copy = copy.deepcopy(hexes_by_label)
            board_copy = curr_board
            hexes_by_label_copy = hexes_by_label

            for hex_info in combination:
                board_copy[hex_info[0]]['owner'] = player_color

            for hex_info in combination:
                hexes_by_label_copy[label].remove(hex_info)

            utility, moves = recursive_min_max_search(
                hexes_by_label_copy, board_copy, me_player_black, remaining_depth - 1, alpha, beta, not max_mode, game_round_number)

            for hex_info in combination:
                board_copy[hex_info[0]]['owner'] = None

            for hex_info in combination:
                hexes_by_label_copy[label].append(hex_info)

            if utility == best_utility and random.random() < 0.5:
                moves = tuple([hex_info[0] for hex_info in combination])
                best_moves = moves

            if max_mode:
                if utility > best_utility:
                    moves = tuple([hex_info[0] for hex_info in combination])

                    best_utility = utility
                    best_moves = moves
                alpha = max(alpha, utility)
            else:
                if utility < best_utility:
                    moves = tuple([hex_info[0] for hex_info in combination])

                    best_utility = utility
                    best_moves = moves
                beta = min(beta, utility)
            if beta <= alpha:
                break

    # if max_mode:
    #     print ("max: best utility: " + str(best_utility) + " best moves: " + str(best_moves))
    # else:
    #     print ("min: best utility: " + str(best_utility) + " best moves: " + str(best_moves))

    return best_utility, best_moves


def init_min_max_search(hexes_by_label, curr_board, me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number):
    # curr_player_black = me_player_black ^ (not max_mode)
    # player_color = 'black' if curr_player_black else 'white'

    if remaining_depth == 0:
        print("error: remaining depth is 0 at init")
    if not max_mode:
        print("error: max mode is false at init")
    # TODO: put the logic to decide whether move is worth or not into external function

    #set number of processes
    number_of_processes = 40

    #generate sub tasks for the processes
    possible_moves = generate_possible_moves(
        hexes_by_label, curr_board, me_player_black, game_round_number)
    random.shuffle(possible_moves)
    # chunk_size = len(possible_moves) // number_of_processes + 1 #TODO experiment with smaller chunk size
    # moves_chunks = [possible_moves[i:i+chunk_size]
    #                 for i in range(0, len(possible_moves), chunk_size)]

    with Pool(processes= number_of_processes) as pool:
        moves = pool.starmap(recursive_min_max_search, [(hex, board, me_player_black, remaining_depth-1, alpha, beta, not max_mode, game_round_number)
                                for hex, board, comb in possible_moves])

    return max(moves, key=lambda x: x[0])

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


def benchmark_count_continuous_neighbors():
    coords = [(0, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    return count_continuous_neighbors(coords)


# Benchmark the function using timeit.timeit()
# execution_time = timeit.timeit(
#     benchmark_count_continuous_neighbors, number=40000)
# print(f"Execution time for 10000 calls: {execution_time:.6f} seconds")
# # Example usage:
# coordinates = [(0, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
# print(count_continuous_neighbors(coordinates))  # Output: 6

# coordinates2 = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
# print(count_continuous_neighbors(coordinates2))  # Output: 2 (since they don't form a continuous cluster)


def are_neighbors(coord1, coord2):
    # Define the six possible relative positions of neighbors in a hexagonal grid
    neighbors = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]

    # Calculate the relative position between the two coordinates
    relative_position = (coord2[0] - coord1[0], coord2[1] - coord1[1])

    # Check if the relative position is one of the six possible neighbors
    return relative_position in neighbors


def count_hexagons_of(curr_board, player_black):
    color = 'black' if player_black else 'white'
    owner_count = 0
    for hex_info in curr_board.values():
        owner = hex_info['owner']
        if owner is color:
            owner_count += 1
    return owner_count

# # Example usage:
# coord1 = (0, 0)
# coord2 = (0, 1)
# print(are_neighbors(coord1, coord2))  # Output: True

# coord3 = (1, 1)
# print(are_neighbors(coord1, coord3))  # Output: False


def evaluate_board_position(curr_board, player_black, game_round_number):
    # calculate the utility of the board position for player player_black
    # own_connected_area = count_connected_area(curr_board, 'black')
    # enemy_connected_area = count_connected_area(curr_board, 'white')
    # if not player_black:
    #     own_connected_area, enemy_connected_area = enemy_connected_area, own_connected_area
    color = 'black' if player_black else 'white'
    enemy_color = 'white' if player_black else 'black'
    own_connected_area = count_connected_area(curr_board, color)
    enemy_connected_area = count_connected_area(curr_board, enemy_color)
    own_total_area = count_hexagons_of(curr_board, player_black)
    enemy_total_area = count_hexagons_of(curr_board, not player_black)
    connected_factor = game_round_number * 0.1

    return (own_connected_area - enemy_connected_area) * connected_factor + own_total_area - enemy_total_area

# fix of the original calculate_connected_areas function
#


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