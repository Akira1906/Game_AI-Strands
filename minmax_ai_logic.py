
"""
This module contains functions for performing a min-max search on a game tree to evaluate possible moves.
"""
import random
import time
from multiprocessing import Pool, cpu_count, Manager, TimeoutError
import tracemalloc
from ai_helper_functions import gen_combinations, generate_promising_moves_with_board, is_promising_move, evaluate_board_position, generate_minimized_promising_moves
import copy

def recursive_min_max_search(hexes_by_label, curr_board, is_me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number):
    """
    Performs a min-max search on the game tree to evaluate possible moves.

    Args:
        hexes_by_label (dict): Available hexes to choose from grouped by label.
        curr_board (dict): Copy of the game board.
        is_me_player_black (bool): Indicates if the max player is black.
        remaining_depth (int): Remaining depth of the search.
        alpha (float): The best value that the maximizing player can guarantee at this level or above.
        beta (float): The best value that the minimizing player can guarantee at this level or above.
        max_mode (bool): Indicates if it's the max player's turn.
        game_round_number (int): The current round number of the game.

    Returns:
        tuple: The utility of the game position and the best moves to make.
    """
    # when the game is over return or the depth is reached return the evaluation of the board
    if remaining_depth == 0 or all(not hex_list for hex_list in hexes_by_label.values()):
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
        if not hex_list:
            continue
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
            #apply move
            for hex_field in move:
                curr_board[hex_field[0]]['owner'] = curr_player_color

            hexes_by_label[label] = [
                hex_field for hex_field in hexes_by_label[label] if hex_field not in move]

            utility, moves = recursive_min_max_search(
                hexes_by_label, curr_board, is_me_player_black, remaining_depth - 1,
                alpha, beta, not max_mode, game_round_number + 1)

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


def start_thread(move, hexes_by_label, curr_board, is_me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number, lock, memory_debug):
    """
    Starts a thread to perform the min-max search.

    Args:
        hexes_by_label (dict): Available hexes to choose from grouped by label.
        curr_board (dict): Copy of the game board.
        is_me_player_black (bool): Indicates if the current player is black.
        remaining_depth (int): Remaining depth of the search.
        alpha (float): The best value that the maximizing player can guarantee at this level or above.
        beta (float): The best value that the minimizing player can guarantee at this level or above.
        max_mode (bool): Indicates if it's the max player's turn.
        game_round_number (int): The current round number of the game.
        lock (Lock): A lock to synchronize access to shared variables.
        memory_debug (bool): Indicates if memory debugging is enabled.

    Returns:
        tuple: The utility of the game position and the peak memory usage (if memory_debug is active).
    """
    if memory_debug:
        tracemalloc.start()

    if not max_mode:  # inversed because we get max_mode as inversed
        with lock:
            local_alpha = alpha.value
            local_beta = float('inf')
    else:
        with lock:
            local_beta = beta.value
            local_alpha = -float('inf')

    # apply the first move
    player_color = 'black' if is_me_player_black else 'white'
    hexes_by_label[move[1]] = [hex_field for hex_field in hexes_by_label[move[1]] if hex_field[0] not in move[0]]
    for hex_coords in move[0]:
        curr_board[hex_coords]['owner'] = player_color


    best_move = recursive_min_max_search(hexes_by_label, curr_board, is_me_player_black,
                                         remaining_depth, local_alpha, local_beta, max_mode, game_round_number)

    # Stop tracing and get the current, peak and cumulative memory usage
    if memory_debug:
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        memory_peak = -1

    # update global alpha and beta
    if not max_mode:  # inversed because we get max_mode as inversed
        with lock:
            alpha.value = max(alpha.value, best_move[0])
    else:
        with lock:
            beta.value = min(beta.value, best_move[0])
    return best_move, memory_peak


def init_min_max_search(hexes_by_label, curr_board, is_me_player_black, game_round_number, timeout):
    """
    Initializes the min-max search.

    Args:
        hexes_by_label (dict): Available hexes to choose from grouped by label.
        curr_board (dict): Copy of the game board.
        is_me_player_black (bool): Indicates if the current player is black.
        game_round_number (int): The current round number of the game.
        timeout (float): The maximum time allowed for the search.

    Returns:
        tuple: The best move to make
    """
    # SETUP the search
    max_mode = True
    memory_debug = False
    start_time = time.time()
    thread_num = cpu_count()

    pure_promising_moves = generate_minimized_promising_moves(
        hexes_by_label, game_round_number)
    print(f"{len(pure_promising_moves)} Pure First Level Moves calculcated")

    # promising_moves = generate_promising_moves_with_board(
    #     hexes_by_label, curr_board, is_me_player_black, game_round_number)
    # random.shuffle(promising_moves)
    # print(f"{len(promising_moves)} First Level Branches will execute in parallel")
    # set(promising_moves)
    # print(f"{len(promising_moves)} First Level Branches after removing duplicates")
    # reduce timeout for extra time buffer
    timeout -= 1
    # start iteration depth: 2
    iteration_depth = 2 - 1
    max_moves = []
    moves = []
    
    while timeout - (time.time() - start_time) > 0:
        local_start_time = time.time()
        curr_timeout = timeout - (time.time() - start_time)
        iteration_depth += 1
        print(
            f"Iteration: {iteration_depth} starts, time limit: {curr_timeout} s")
        with Manager() as manager:
            alpha = manager.Value('d', -float('inf'))
            beta = manager.Value('d', float('inf'))
            lock = manager.Lock()

            with Pool(processes=thread_num) as pool:

                results = [pool.apply_async(start_thread,
                                            (move, hexes_by_label, curr_board, is_me_player_black, iteration_depth - 1, alpha,
                                             beta, not max_mode, game_round_number + 1, lock, memory_debug)) for move in pure_promising_moves]

                time_to_wait = curr_timeout  # initial time to wait
                for result in results:
                    try:
                        # wait for up to time_to_wait seconds
                        result.get(time_to_wait)
                    except TimeoutError:
                        pass

                    # how much time has expired since we began waiting?
                    t = time.time() - local_start_time
                    time_to_wait = curr_timeout - t
                    if time_to_wait < 0:
                        time_to_wait = 0
                pool.terminate()  # all processes, busy or idle, will be terminated

                results = [result.get()
                           for result in results if result.ready()]
                if results:
                    moves, memory_usages = zip(*results)
                    max_moves.append(max(moves, key=lambda x: x[0]))

    if memory_debug:
        print(f"Memory usage of all processes: {sum(memory_usages)/ 10**6}MB ")
        print(
            f"Average memory usage of processes: {(sum(memory_usages)/ len(memory_usages))/ 10**6}MB")
    if not max_moves:
        return None
    return max(max_moves, key=lambda x: x[0])[1]
