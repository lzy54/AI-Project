"""
Bridge Crossing Problem
Author: Zhuoyang Li

This program solves the bridge crossing problem using Uniform Cost Search (UCS) and A* algorithms.
"""

import sys
import heapq
from itertools import combinations

# Robot crossing times
robot_times = {
    'A': 1,
    'B': 2,
    'C': 5,
    'D': 10
}

# List of all robots
robots = ['A', 'B', 'C', 'D']

# State class
class State:
    # Constructor
    # positions: dictionary of robot positions (True: start side, False: end side)
    # time: time spent to reach this state
    def __init__(self, positions, time):
        self.positions = positions 
        self.time = time 
        self.parent = None # parent state
        self.action = None # action that led to this state
        
    # for priority queue comparison
    def __lt__(self, other):
        return self.time < other.time
    
    def __eq__(self, other):
        return self.positions == other.positions
    
    def __hash__(self):
        return hash(tuple(self.positions))
    
    
#Helper functions
def calculate_depth(state):
    depth = 0
    while state.parent is not None:
        state = state.parent
        depth += 1
    return depth

def reconstruct_path(state):
    path = []
    while state.parent is not None:
        path.append(state.action)
        state = state.parent
    path.reverse()
    return path

def is_valid(state):
    return all(not pos for pos in state.positions.values())

def apply_action(state, action):
    new_positions = state.positions.copy()
    move_direction, moving_robots = action
    current_time = state.time
    
    # False: end side, True: start side
    if move_direction == '->':
        for robot in moving_robots:
            new_positions[robot] = False
        new_positions['P'] = False
    else:
        for robot in moving_robots:
            new_positions[robot] = True
        new_positions['P'] = True
        
    # Velocity = speed of the slowest robot
    move_time = max([robot_times[robot] for robot in moving_robots])
    new_time = current_time + move_time
    
    new_state = State(new_positions, new_time)
    new_state.parent = state
    new_state.action = action
    
    return new_state

def generate_actions(state):
    actions = []
    positions = state.positions
    pack_side = positions['P']
    robots_on_pack_side = [robot for robot in robots if positions[robot] == pack_side]
    
    if pack_side:
        #power is on the start side
        #enumerate all possible combinations of robots(2 robots /pair) on the pack side
        for pair in combinations(robots_on_pack_side, 2):
            actions.append(('->', pair))
        #power is on the end side
        #generate actions for one robot to gp back to the start side
    else:
        for robot in robots_on_pack_side:
            actions.append(('<-', [robot]))
    return actions

# ex: initialize("ACDP", "B")
def initialize(start_side_str, end_side_str):
    positions = {}
    for item in ["A", "B", "C", "D", "P"]:
        if item in start_side_str:
            positions[item] = True
        elif item in end_side_str:
            positions[item] = False
        else:
            print(f"Invalid input: {item} not found in start or end side")
            sys.exit(1)
            
    return State(positions, time=0)

def parse_input():
    # Check for valid input
    if len(sys.argv) != 4:
        print("Usage: python bridge_crossing.py <start_side> <end_side> <algorithm>")
        sys.exit(1)
        
    start_side = sys.argv[1]
    end_side = sys.argv[2]
    algorithm = sys.argv[3]
        
    if algorithm not in ['UC','Astar']:
        print("Invalid algorithm. Choose one of: bfs, dfs, ucs")
        sys.exit(1)
        
    all_chars = start_side + end_side
    if sorted(all_chars) != sorted("ABCDP"):
        print("Invalid input. Make sure all robots and power are included")
        sys.exit(1)
   
    # Check if power is alone         
    if ('P' in start_side and len(start_side) == 1) or ('P' in end_side and len(end_side) == 1):
        print("Invalid input. Power cannot cross alone")
        sys.exit(1)
            
    return start_side, end_side, algorithm

# UCS algorithm
def UCS(start_state):
    frontier = []
    # add start state to the frontier
    heapq.heappush(frontier, (start_state.time, start_state))
    # set of explored states
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        # pop the state with the lowest cost
        current_cost, current_state = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # print the current state
        parent_positions = current_state.parent.positions if current_state.parent is not None else None
        print(f"Expanded Nodes: {current_state.positions},\n  Parent State: {parent_positions},\n  time: {current_state.time}, depth: {calculate_depth(current_state)}")
        
        # check if the current state is the goal state
        if is_valid(current_state):
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Solution Cost: {current_state.time}")
            return reconstruct_path(current_state)
        
        explored.add(current_state)
        
        # generate actions and apply them to the current state
        for action in generate_actions(current_state):
            new_state = apply_action(current_state, action)
            if new_state not in explored:
                heapq.heappush(frontier, (new_state.time, new_state))
    
    print("No solution found")
    return None

# A* algorithm
# heuristic function: number of robots on the start side
# heuristic suggests that everyone moves at the same speed as the fastest robot(1 min).
def heuristic(state):
    robots_on_start_side = [robot for robot in robots if state.positions[robot]]
    return len(robots_on_start_side) * robot_times['A']

def Astar(start_state):
    frontier = []
    heapq.heappush(frontier, (start_state.time + heuristic(start_state), start_state))
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        _, current_state = heapq.heappop(frontier)
        nodes_expanded += 1
        
        parent_positions = current_state.parent.positions if current_state.parent is not None else None
        print(f"Expanded Nodes: {current_state.positions},\n  Parent State: {parent_positions},\n  time: {current_state.time}, depth: {calculate_depth(current_state)}")
        
        if is_valid(current_state):
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Solution Cost: {current_state.time}")
            return reconstruct_path(current_state)
        
        explored.add(current_state)
        
        for action in generate_actions(current_state):
            new_state = apply_action(current_state, action)
            if new_state not in explored:
                total_time = new_state.time + heuristic(new_state)
                heapq.heappush(frontier, (new_state.time + heuristic(new_state), new_state))
    print("No solution found")
    return None

# Main function
def main():
    start_side_str, end_side_str, algorithm = parse_input()
    start_state = initialize(start_side_str, end_side_str)
    
    # Run the selected algorithm
    if algorithm == 'UC':
        print("Running UCS algorithm: \n")
        path = UCS(start_state)
    elif algorithm == 'Astar':
        print("Running A* algorithm: \n")
        path = Astar(start_state)
    else:
        print("Invalid algorithm")
        sys.exit(1)
        
    if path is not None:
        print("Solution Path: ")
        for action in path:
            direction, robots_moved = action
            robots_str = ', '.join(robots_moved)
            if direction == '->':
                print(f"Move {robots_str} to the end side")
            else:
                print(f"Move {robots_str} back to the start side")
    else:
        print("No solution found")

if __name__ == "__main__":
    main()