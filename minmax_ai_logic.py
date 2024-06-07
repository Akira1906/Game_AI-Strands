
import os
import random
import time
from multiprocessing import Pool, cpu_count, Manager, TimeoutError
import tracemalloc
import psutil
from ai_helper_functions import gen_combinations, generate_promising_moves_with_board, is_promising_move, evaluate_board_position


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
    # when the game is over
    if hexes_by_label == []:
        return 0, []

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

            hexes_by_label[label] = [
                hex_field for hex_field in hexes_by_label[label] if hex_field not in move]

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


def start_thread(hexes_by_label, curr_board, is_me_player_black, remaining_depth, alpha, beta, max_mode, game_round_number, lock, memory_debug):
    if (memory_debug):
        tracemalloc.start()

    with lock:
        local_alpha = alpha.value
    local_beta = float('inf')
    # local_alpha = -float('inf')
    best_move = recursive_min_max_search(hexes_by_label, curr_board, is_me_player_black,
                                         remaining_depth, local_alpha, local_beta, max_mode, game_round_number)
    # Stop tracing and get the current, peak and cumulative memory usage
    if (memory_debug):
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        memory_peak = -1
    # print(f"Peak memory usage is {peak / 10**6}MB")
    # update global alpha and beta
    with lock:
        if not max_mode:  # inversed because we get max_mode as inversed
            alpha.value = max(alpha.value, best_move[0])
        else:
            beta.value = min(beta.value, best_move[0])
    return best_move, memory_peak


def init_min_max_search(hexes_by_label, curr_board, is_me_player_black, game_round_number):
    max_mode = True
    memory_debug = False
    start_time = time.time()
    if not max_mode:
        print("error: max mode is false at init")

    # set number of processes
    number_of_processes = cpu_count()
    # print("number of processes: " + str(number_of_processes))
    # generate sub tasks for the processes
    # print all the variables at this point for debugging
    # print("Starting min max search with depth: " + str(remaining_depth) + " and max mode: " + str(max_mode) + " and game round number: " + str(game_round_number))
    # print("hexes by label: " + str(sys.getsizeof(hexes_by_label)))
    # print("current board: " + str(sys.getsizeof(curr_board)))
    # print("is me player black: " + str(is_me_player_black))
    promising_moves = generate_promising_moves_with_board(
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

    # iterative deepening TODO: add game won detection
    TIMEOUT = 9
    iteration_depth = 1
    max_moves = []
    while TIMEOUT - (time.time() - start_time) > 0:
        local_start_time = time.time()
        curr_timeout = TIMEOUT - (time.time() - start_time)
        iteration_depth += 1
        print(
            f"Iteration depth: {iteration_depth}, time left: {curr_timeout} seconds")
        # Create a Manager object
        with Manager() as manager:
            # Create Value and Lock objects using the Manager
            # 'd' indicates a double precision float
            alpha = manager.Value('d', -float('inf'))
            beta = manager.Value('d', float('inf'))
            lock = manager.Lock()

            with Pool(processes=number_of_processes) as pool:
                results = [pool.apply_async(start_thread,
                                            (hex, board, is_me_player_black, iteration_depth - 1, alpha,
                                             beta, not max_mode, game_round_number + 1, lock, memory_debug)) for hex, board in promising_moves]

                time_to_wait = curr_timeout  # initial time to wait
                for i, result in enumerate(results):
                    try:
                        # wait for up to time_to_wait seconds
                        return_value = result.get(time_to_wait)
                    except TimeoutError:
                        pass
                        # print('Timeout for v = ', i)
                    # else:
                        # print(f'Return value for v = {i} is {return_value}')
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
    # print(f"max moves: {max_moves}")
    # print(f"max max moves: {max(max_moves, key=lambda x: x[0])}")
    if not max_moves:
        return None
    return max(max_moves, key=lambda x: x[0])


def monitor_memory(interval=10):
    while True:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory Usage: RSS={memory_info.rss / (1024 * 1024):.2f} MB")
        time.sleep(interval)
