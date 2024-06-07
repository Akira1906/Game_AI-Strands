import itertools
import copy
import math
import os
import random
import sys
import time
import timeit
import concurrent.futures
from multiprocessing import Process, freeze_support, set_start_method, Pool, cpu_count, Value, Lock, Manager, TimeoutError
import threading
import asyncio
from memory_profiler import profile
import tracemalloc
import psutil
from ai_helper_functions import gen_combinations, is_promising_move, evaluate_board_position, generate_promising_moves, count_connected_area
from abc import ABC, abstractmethod
from collections import defaultdict
import math

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node): #TODO: fix this, have a look
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward()
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)

class StrandsBoard:
    def __init__(self, remaining_hexes, hex_board, is_me_player_black, game_round_number, terminal = False, winner = None, my_turn = True):
        self.remaining_hexes = remaining_hexes
        self.hex_board = hex_board
        self.is_me_player_black = is_me_player_black
        self.game_round_number = game_round_number
        self.terminal = terminal
        self.winner = winner
        self.my_turn = my_turn

    def find_children(self):
        if self.is_terminal():
            return []
        return {
            self.make_move(move) for move in generate_promising_moves(self.remaining_hexes, self.hex_board, self.is_me_player_black, self.game_round_number)
        }

    def find_random_child(self):
        if self.is_terminal():
            return None
        promising_moves = generate_promising_moves(self.remaining_hexes, self.hex_board, self.is_me_player_black, self.game_round_number)
        if promising_moves == []:
            return None
        else:
            return self.make_move(random.choice(promising_moves))

    def reward(self):
        #give positive reward if the player has won
        if not self.is_terminal():
            raise RuntimeError("reward called on nonterminal board")

        if (self.winner and self.is_me_player_black) or (not self.winner and not self.is_me_player_black):
            return 1
        elif self.winner is None:
            return 0.5
        else:
            return 0
        
        
        
        # return evaluate_board_position(self.curr_board, self.is_me_player_black, self.game_round_number)

    def is_terminal(self):
        return self.terminal
    
    def make_move(self, move):
        player_color = 'black' if ((self.is_me_player_black and self.my_turn) or (not self.is_me_player_black and not self.my_turn)) else 'white'
        label = move[0][1]["label"]
        hex_board = copy.deepcopy(self.hex_board)
        remaining_hexes = copy.deepcopy(self.remaining_hexes)
        for hex_info in move:
            hex_board[hex_info[0]]['owner'] = player_color

        remaining_hexes[label] = [
            hex_field for hex_field in remaining_hexes[label] if hex_field not in move]
        my_turn = not self.my_turn
        terminal = remaining_hexes[6] == [] and remaining_hexes[5] == [] and remaining_hexes[3] == [] and remaining_hexes[2] == [] and remaining_hexes[1] == []
        winner = None
        if terminal:
            winner =_find_winner(self.hex_board)
        return StrandsBoard(remaining_hexes, hex_board, self.is_me_player_black, self.game_round_number + 1, terminal, winner, my_turn)
        
        
# True if black wins, False if white wins, None if draw
def _find_winner(hex_board):
    black_connected_area = count_connected_area(hex_board, 'black')
    white_connected_area = count_connected_area(hex_board, 'white')
    if black_connected_area > white_connected_area:
        return True
    elif black_connected_area < white_connected_area:
        return False
    else:
        return None

def init_mcts_search(hexes_by_label, hex_board, is_me_player_black, game_round_number):
    tree = MCTS()
    board = StrandsBoard(hexes_by_label, hex_board, is_me_player_black, game_round_number)
    if board.is_terminal():
        return []
    for _ in range(10):
        tree.do_rollout(board)
    board = tree.choose(board)
    print(board)
    
    return []


def mcts_iteration(hexes_by_label, curr_board, is_me_player_black, game_round_number):
    for i in range(1000):
    
    # MCTS
        # select_next_node_to_expand()
    # 1. Selection
    # 2. Expansion
    # 3. Simulation
    # 4. Backpropagation
    # 5. Selection
        pass
