# Technical Considerations for the Python Implementation

While working with the original game code, I encountered some minor issues that needed to be addressed before I could implement the Game Search algorithms.

## Minor Bugs in the Provided Source Code

Several minor bugs in the original program needed fixing before the project could proceed:

- **Pygame Initialization**: Pygame was initialized twice at the beginning.
- **Faulty Implementation of `calculate_connected_areas(color)`**: The function to calculate connected areas did not work correctly and caused issues with the game code. The fix is straightforward and related to the format of `color`.

```python
# fix of the original calculate_connected_areas function
# usage color = 2 = 'white' / 1 = 'black'
def calculate_connected_areas(color):
    # Calculates the largest connected area of hexes of the specified color.
    if color == 1:
        color = 'black'
    elif color == 2:
        color = 'white'
    else:
        print("Error: undefined color")

    def dfs(row, col, visited):
        if (row, col) in visited or not (row, col) in hexagon_board or hexagon_board[(row, col)]['owner'] != color:
            return 0
        visited.add((row, col))
        count = 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            next_row, next_col = row + dr, col + dc
            count += dfs(next_row, next_col, visited)
        return count

    visited = set()
    max_area = 0
    for (row, col), info in hexagon_board.items():
        if (row, col) not in visited and info['owner'] == color:
            area = dfs(row, col, visited)
            if area > max_area:
                max_area = area
    return max_area
```

## Fixes Required for Multithreading in the Original Source Code

The provided code is not suitable for multithreading because each new thread spawns a new instance of the Pygame gameboard. To circumvent this issue, two simple modifications to the provided code are required, which do not alter any game behavior.

### Initialize Game in a Function Instead of at the Root

This issue can be easily avoided by defining a new function that includes all the relevant definitions for starting up the game. Static variables are left out of the function as they do not cause any issues.

```python
# Define static variables and hexagon properties
BG_COLOR = (30, 30, 30)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
HEX_SIZE = 30
HEX_BORDER = 2
PIECE_RADIUS = int(HEX_SIZE * 0.8)

# Initialize global variables for Pygame
screen = None
font = None
screen_info = None
screen_width = None
screen_height = None
WIDTH = None
HEIGHT = None
# Initialize game state variables
hexagon_board = {}
selected_counts = {}
turn_ended = False
max_selected_counts = {}
initial_counts = {}  # Store initial counts for each label

def initialize_pygame():
    global screen, font, screen_info, screen_width, screen_height, WIDTH, HEIGHT
    
    pygame.init()

    # Set environment variable to ensure the window opens at the center of the screen
    os.environ['SDL_VIDEO_CENTERED'] = '1'

    # Get current screen resolution
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h

    # Create a window
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set
```

## Utilization of My Game-Search Code

The project consists of three additional files which should be in the same directory as the `main_r6.py` file: `mcsts_ai_logic.py`, `minimax_ai_logic.py`, and `ai_helper_functions.py`. The tests in `test_ai_helper_functions.py` are not necessarily required. To ensure the AI works correctly, additional imports are needed in the `main_r6.py` file.

```python
from minmax_ai_logic import init_min_max_search
from mcts_ai_logic import init_mcts_search
```

As required in the project description, I implemented my own version of select_hexes_by_random(hexes_by_label, curr_round). My implementation additionally requires one extra function argument: current_turn. This variable is available where select_hexes_by_random is usually executed, so this should not present any issues. In my project code, I call my new function select_hexes_by_ai(hexes_by_label, current_round, current_turn). It is implemented as follows. The inclusion of the code is for demonstration purposes only; the actual implementation should be taken from the project files as it might differ slightly.

Regarding utilization, it is worth mentioning that the mode of the AI can be changed to either of the two algorithms by setting the mode variable to 'MINMAX' or 'MCTS' respectively.

```python
def select_hexes_by_ai(hexes_by_label, curr_round, curr_turn):
    import copy
    import time
    selected_hexes = []
    if curr_round == 1:
        # Special handling for the first round: only select one hexagon labeled as 2.
        if 2 in hexes_by_label and any(not hex_info['selected'] for _, hex_info in hexes_by_label[2]):
            available_hexes = [
                (pos, hex_info) for pos, hex_info in hexes_by_label[2] if not hex_info['selected']]
            if available_hexes:
                selected_hexes.append(random.choice(available_hexes))
    else:
        # Calculate remaining unselected hexes for each label and choose only those labels with remaining hexes
        available_labels = {
            label: hexes
            for label, hexes in hexes_by_label.items()
            if any(not hex_info['selected'] for _, hex_info in hexes)
        }
        if available_labels:
            is_me_player_black = curr_turn == 'black'

            # minimize the game board object and hexagon list object
            board_copy = copy.deepcopy(hexagon_board)
            board_copy = {pos: {
                'label': info['label'], 'owner': info['owner']} for pos, info in board_copy.items()}
            hexes_by_label_copy = copy.deepcopy(hexes_by_label)
            hexes_by_label_copy = {
                label: [(coord, {'label': details['label'], 'owner': details['owner']})for coord, details in hex_list] for label, hex_list in hexes_by_label_copy.items()
            }

            mode = 'MINMAX' # 'MCTS' or 'MINMAX'

            local_start_time = time.time()
            TIMEOUT = 15
            if mode == "MINMAX":
                best_move = init_min_max_search(hexes_by_label_copy, board_copy, is_me_player_black, curr_round, TIMEOUT)
                best_move = best_move[1]
            elif mode == "MCTS":
                best_move = init_mcts_search(hexes_by_label_copy, board_copy, is_me_player_black, curr_round, TIMEOUT)
            end_time = time.time()

            print("Time taken: " + str(end_time - local_start_time)+"s")
            print(mode + ": choose best move -> " + str(best_move))
            if best_move is not None and len(best_move) > 0:
                selected_hexes.extend([(hex, hexagon_board[hex])
                                  for hex in hexagon_board.keys() if hex in best_move])
            # select random choice when there is not decision made
            else:
                selected_label = random.choice(list(available_labels.keys()))
                available_hexes = [
                    (pos, hex_info) for pos, hex_info in available_labels[selected_label] if not hex_info['selected']]
                # Determine the number of hexagons to select based on their labels.
                n = selected_label

                # Randomly select n hexes, select all remaining hexes if fewer than n are available
                if len(available_hexes) > n:
                    selected_hexes.extend(random.sample(available_hexes, n))
                else:
                    selected_hexes.extend(available_hexes)
        print(curr_round)

        
    return selected_hexes
```

## Algorithm Setup
These are the configurations possible for the Algorithms:

- Choose algorithm
- Number of processes utilized
- Execution time limit
```python
# mcts_ai_logic.py
def init_mcts_search(hexes_by_label, hex_board, is_me_player_black, game_round_number, timeout):
    #...
    number_of_processes = 1
    
# minmax_ai_logic.py
def init_min_max_search(hexes_by_label, curr_board, is_me_player_black, game_round_number, timeout):
# ...
process_num = cpu_count()

# main.py
def select_hexes_by_ai(hexes_by_label, curr_round, curr_turn):
#...
MODE = 'MINMAX' # 'MCTS' or 'MINMAX'
TIMEOUT = 30
```