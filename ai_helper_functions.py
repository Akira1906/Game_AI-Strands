import copy
import random
import itertools


def generate_promising_moves_with_board(hexes_by_label, curr_board, is_player_black, game_round_number):
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


def generate_random_promising_move(hexes_by_label, game_round_number):
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


def is_promising_move(len_fields_of_label, move, game_round_number, label):
    continuous_neighbors = count_continuous_neighbors(
        [hex_info[0] for hex_info in move])
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
    else:
        return True
    return True


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
