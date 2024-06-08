import itertools
import copy
import math
import os
import random
import sys
import time
import timeit
import concurrent.futures
from multiprocessing import Process, freeze_support, set_start_method, Pool, cpu_count, Value, Lock, Manager, TimeoutError, Process
import threading
import asyncio
from memory_profiler import profile
import tracemalloc
import psutil
from ai_helper_functions import gen_combinations, is_promising_move, evaluate_board_position, generate_promising_moves, count_connected_area, generate_random_promising_move
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

    def choose(self, node):  # TODO: fix this, have a look
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
        self._expand(leaf)  # slow
        reward = self._simulate(leaf)  # slow
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
    def __init__(self, remaining_hexes, hex_board, is_me_player_black, game_round_number, terminal=False, winner=None, my_turn=True):
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
            self.make_move(move) for move in generate_promising_moves(self.remaining_hexes, self.game_round_number)
        }

    def find_random_child(self):
        if self.is_terminal():
            return None
        random_promising_move = generate_random_promising_move(
            self.remaining_hexes, self.game_round_number)
        if random_promising_move == []:
            return None
        else:
            return self.make_move(random_promising_move)

    def reward(self):
        # give positive reward if the player has won
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
        player_color = 'black' if ((self.is_me_player_black and self.my_turn) or (
            not self.is_me_player_black and not self.my_turn)) else 'white'
        label = move[0][1]["label"]
        hex_board = copy.deepcopy(self.hex_board)
        remaining_hexes = copy.deepcopy(self.remaining_hexes)
        for hex_info in move:
            hex_board[hex_info[0]]['owner'] = player_color

        remaining_hexes[label] = [
            hex_field for hex_field in remaining_hexes[label] if hex_field not in move]
        my_turn = not self.my_turn
        # check if every list in remaining_hexes is empty
        terminal = all([len(hex_list) == 0 for hex_list in remaining_hexes.values(
        )]) or remaining_hexes == []
        winner = None
        if terminal:
            winner = _find_winner(self.hex_board)
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


def start_mcts_thread(hexes_by_label, hex_board, is_me_player_black, game_round_number, time_limit, start_time, return_dict, index):
    tree = MCTS()
    board = StrandsBoard(hexes_by_label, hex_board,
                         is_me_player_black, game_round_number)
    if board.is_terminal():
        return []

    while time.time() - start_time < time_limit - 1:
        tree.do_rollout(board)
    return_dict[index] = (tree, board)


def init_mcts_search(hexes_by_label, hex_board, is_me_player_black, game_round_number, timeout):
    start_time = time.time()
    # root parallelization
    # create multiple trees in parallel, later compare the nodes and choose the best one
    number_of_threads = 1
    timeout -= 1

    processes = []
    manager = Manager()
    return_dict = manager.dict()

    for i in range(number_of_threads):
        process = Process(
            target=start_mcts_thread,
            args=(hexes_by_label, hex_board, is_me_player_black,
                  game_round_number, timeout, start_time, return_dict, i)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    # aggregate results from all threads
    aggregated_tree = MCTS()
    for tree, board in return_dict.values():
        for node in tree.N:
            aggregated_tree.N[node] += tree.N[node]
            aggregated_tree.Q[node] += tree.Q[node]
        for node in tree.children:
            if node not in aggregated_tree.children:
                aggregated_tree.children[node] = tree.children[node]
    best_board = aggregated_tree.choose(StrandsBoard(
        hexes_by_label, hex_board, is_me_player_black, game_round_number))

    # difference between hex_board and board.hex_board
    moves = []
    for coords, info in hex_board.items():
        if info['owner'] != best_board.hex_board[coords]['owner']:
            moves.append(coords)

    return moves
