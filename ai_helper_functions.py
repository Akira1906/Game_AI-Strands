import copy
import random
import itertools


def is_promising_move_min(len_fields_of_label, move, game_round_number, label):
    """
    Determines if a move is promising based on the given parameters.

    Args:
        len_fields_of_label (int): The length of fields of the given label.
        move (list): A list of hex_info representing the move.
        game_round_number (int): The current round number of the game.
        label (int): The label to check for.

    Returns:
        bool: True if the move is promising, False otherwise.
    """
    # continuous_neighbors = count_continuous_neighbors(
        # [hex_info[0] for hex_info in move])
    early_start_phase = 4
    start_phase = 10
    mid_phase = 20
    end_phase = 34

    if game_round_number < early_start_phase:
        if label in [2, 3]:
            return False

    # elif game_round_number < start_phase:
    #     if label in [2, 3] and continuous_neighbors < label:
    #         return False
    #     elif label == 5 and continuous_neighbors < 3:
    #         return False

    # elif game_round_number < mid_phase:
    #     if label == 2 and len_fields_of_label > 10:
    #         if continuous_neighbors < 2:
    #             return False

    #     elif label == 3 and len_fields_of_label > 10:
    #         if continuous_neighbors < 3:
    #             return False
    # else:
    #     return True
    return True

def is_promising_move(len_fields_of_label, move, game_round_number, label):
    """
    Determines if a move is promising based on the given parameters.

    Args:
        len_fields_of_label (int): The length of fields of the given label.
        move (list): A list of hex_info representing the move.
        game_round_number (int): The current round number of the game.
        label (int): The label to check for.

    Returns:
        bool: True if the move is promising, False otherwise.
    """
    continuous_neighbors = count_continuous_neighbors(
        [hex_info[0] for hex_info in move])
    early_start_phase = 4
    start_phase = 10
    mid_phase = 20
    end_phase = 34

    if label == 5 and len_fields_of_label > 10:
        if continuous_neighbors < 3:
            return False

    if label == 3 and len_fields_of_label > 15:
        if continuous_neighbors < 2:
            return False
    if game_round_number < early_start_phase:
        if label in [2, 3] and continuous_neighbors < label:
            return False
        if label == 5 and continuous_neighbors < 4:
            return False

    elif game_round_number < start_phase:
        if label in [2, 3] and continuous_neighbors < label:
            return False
        if label == 5 and continuous_neighbors < 3:
            return False

    # elif game_round_number < mid_phase:
    #     if label == 2 and len_fields_of_label > 10:
    #         if continuous_neighbors < 2:
    #             return False

    #     elif label == 3 and len_fields_of_label > 10:
    #         if continuous_neighbors < 3:
    #             return False

    return True


def is_promising_move_pure(len_fields_of_label, move, game_round_number, label):
    """
    Determines if a move is promising based on the given parameters.

    Args:
        len_fields_of_label (int): The length of fields of the given label.
        move (list): A list of hex_info representing the move.
        game_round_number (int): The current round number of the game.
        label (int): The label to check for.

    Returns:
        bool: True if the move is promising, False otherwise.
    """
    continuous_neighbors = count_continuous_neighbors(move)
    early_start_phase = 4
    start_phase = 10
    mid_phase = 20
    end_phase = 34

    if game_round_number < early_start_phase:
        if label in [2, 3]:
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
            if continuous_neighbors < 2:
                return False
    else:
        return True
    return True


def evaluate_board_position(curr_board, is_me_player_black, game_round_number):
    """
    Evaluates the board position based on the current state of the board.

    Parameters:
    curr_board (list): The current state of the board.
    player_black (bool): True if the player is black, False if the player is white.
    game_round_number (int): The current round number of the game.

    Returns:
    float: The evaluation score of the board position.
    """
    color = 'black' if is_me_player_black else 'white'
    enemy_color = 'white' if is_me_player_black else 'black'
    own_connected_area = my_count_connected_area(curr_board, color)
    enemy_connected_area = my_count_connected_area(curr_board, enemy_color)
    own_total_area = count_hexagons_of(curr_board, color)
    enemy_total_area = count_hexagons_of(curr_board, enemy_color)
    connected_factor = 1#game_round_number * 0.05

    return (own_connected_area - enemy_connected_area) * connected_factor


def generate_promising_moves_with_board(hexes_by_label, curr_board, is_player_black, game_round_number):
    """Generate a list of promising moves with the given board state.

    This function takes in the hexes_by_label dictionary, the current board state, 
    a boolean indicating whether the player is black or white, and the game round number.
    It generates a list of promising moves based on the given parameters filtering the moves
    utilizing the is_promising_move function.

    Args:
        hexes_by_label (dict): A dictionary mapping labels to lists of hexes.
        curr_board (dict): The current board state.
        is_player_black (bool): A boolean indicating whether the player is black.
        game_round_number (int): The current game round number.

    Returns:
        list: A list of promising moves, where each move is represented as a tuple 
        containing the updated hexes_by_label dictionary and the updated board state.
    """
    player_color = 'black' if is_player_black else 'white'
    promising_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label = int(label)

        if len(hex_list) < label:
            if len(hex_list) == 0:
                move_combinations = []
            else:
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

            hexes_by_label_copy[label] = [
                hex_field for hex_field in hexes_by_label_copy[label] if hex_field not in move]

            promising_moves.append(
                (hexes_by_label_copy, board_copy))
    return promising_moves


def generate_promising_moves(hexes_by_label, game_round_number):
    """
    Generates a list of promising moves based on the given hexes and game round number.
    Filters the moves utilizing the is_promising_move function.

    Args:
        hexes_by_label (dict): A dictionary containing hexes grouped by their labels.
        game_round_number (int): The current round number of the game.

    Returns:
        list: A list of promising moves.

    """
    promising_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label = int(label)

        if len(hex_list) < label:
            if len(hex_list) == 0:
                move_combinations = []
            else:
                move_combinations = [hex_list]
        else:
            move_combinations = gen_combinations(hex_list, label)

        for move in move_combinations:
            if not is_promising_move(len(hexes_by_label[label]), move, game_round_number, label):
                continue
            promising_moves.append(move)
    return promising_moves


def generate_minimized_promising_moves(hexes_by_label, game_round_number):
    """
    Generates a list of promising moves based on the given hexes and game round number.
    Filters the moves utilizing the is_promising_move function.

    Args:
        hexes_by_label (dict): A dictionary containing hexes grouped by their labels.
        game_round_number (int): The current round number of the game.

    Returns:
        list: A list of promising moves.

    """
    promising_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = [h[0] for h in hexes_by_label[label]]
        label = int(label)

        if len(hex_list) < label:
            if len(hex_list) == 0:
                move_combinations = []
            else:
                move_combinations = [hex_list]
        else:
            move_combinations = gen_combinations(hex_list, label)

        for move in move_combinations:
            if not is_promising_move_pure(len(hexes_by_label[label]), move, game_round_number, label):
                continue
            promising_moves.append((move, label))
    return promising_moves


def generate_random_promising_move(hexes_by_label, game_round_number):
    """
    Generates a random promising move based on the given hexes and game round number.
    Utilizes the is_promising_move function to filter the moves and returns a randomly selected move.
    The function was designed to speed up random move generation.

    Args:
        hexes_by_label (dict): A dictionary containing hexes grouped by label.
        game_round_number (int): The current round number of the game.

    Returns:
        list: A randomly selected promising move.
    """
    random_promising_moves = []
    for label in list(hexes_by_label.keys()):
        hex_list = hexes_by_label[label]
        label = int(label)

        if len(hex_list) < label:
            if len(hex_list) == 0:
                pass
            elif is_promising_move(len(hexes_by_label[label]), hex_list, game_round_number, label):

                random_promising_moves.append(hex_list)
        else:
            for move in gen_random_combinations(hex_list, label):
                if is_promising_move(len(hexes_by_label[label]), move, game_round_number, label):
                    random_promising_moves.append(move)
                    break
    return random.choice(random_promising_moves)


def gen_combinations(iterable, r):
    """
    Generate combinations of length r from an iterable.

    Args:
        iterable: An iterable object from which combinations are generated.
        r: The length of each combination.

    Yields:
        A tuple representing a combination of length r.
    """
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


def gen_random_combinations(iterable, r):
    """
    Generate random combinations of length r from the given iterable.
    This function was designed to enable efficient random move generation.

    Args:
        iterable (iterable): The input iterable from which combinations are generated.
        r (int): The length of each combination.

    Yields:
        tuple: A tuple representing a random combination of length r.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return

    # Generate permutation indices in a pseudorandom order
    indices = list(range(r))
    max_index = len(list(itertools.combinations(range(n), r)))
    order = list(range(max_index))
    random.shuffle(order)

    for index in order:
        # Generate the specific permutation from the permutation index
        indices = nth_combination(index, range(n), r)
        yield tuple(pool[i] for i in indices)


def nth_combination(index, iterable, r):
    'Equivalent to list(itertools.combinations(iterable, r))[index]'
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = 1
    k = min(r, n-r)
    for i in range(1, k+1):
        c = c * (n - k + i) // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    while r:
        c, n, r = c*r//n, n-1, r-1
        while index >= c:
            index -= c
            c, n = c*(n-r+1)//n, n-1
            if n == 0:
                break
        result.append(pool[-1-n])
    return tuple(result)


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


def count_continuous_neighbors_and_continuous_length(coordinates):
    """
    Counts the size and dimension of the largest cluster of continuous neighbors in a given set of coordinates.

    Args:
        coordinates (list): A list of coordinate tuples representing the positions of points.

    Returns:
        tuple: A tuple containing the size of the largest cluster and the dimension of the largest cluster.

    Example:
        coordinates = [(0, 0), (0, 1), (1, 1), (2, 2), (2, 3)]
        count_continuous_neighbors_and_continuous_length(coordinates)
        # Output: (3, 2)
    """
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
    """
    Counts the size of the largest cluster of continuous neighbors in a set of coordinates.

    Parameters:
    - coordinates (list): A list of coordinate tuples representing points in a grid.

    Returns:
    - max_cluster_size (int): The size of the largest cluster of continuous neighbors.

    Example:
        coordinates = [(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)]
        count_continuous_neighbors(coordinates)
    3
    """

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


def my_count_connected_area(curr_board, player_color):
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
