# Bridge Crossing Problem

**Author:** Zhuoyang Li

## Overview

This project provides a solution to the classic Bridge Crossing Problem using two search algorithms: Uniform Cost Search (UCS) and A* Search. The problem involves four robots with different crossing times trying to cross a bridge with a shared power pack. The challenge is to find the optimal sequence of crossings that minimizes the total time.

## Problem Description

Four robots (`A`, `B`, `C`, `D`) need to cross a bridge, but they can only cross while sharing a power pack (`P`). The time each robot takes to cross is different:

- **Robot A**: 1 minute
- **Robot B**: 2 minutes
- **Robot C**: 5 minutes
- **Robot D**: 10 minutes

At most two robots can cross the bridge at once, and they must have the power pack with them. The goal is to minimize the total time for all robots to reach the other side of the bridge.

## Algorithms Implemented

### Uniform Cost Search (UCS)
UCS is used to find the path with the minimum cost (in terms of time) to get all robots across the bridge. It expands the least costly node first and guarantees an optimal solution.

### A* Search
A* search uses a heuristic to guide the search process towards the goal more efficiently. In this implementation, the heuristic is the number of robots still on the start side multiplied by the time of the fastest robot (`A`).

## Program Structure

- **State Class**: Represents the current positions of all robots and the total time spent so far.
- **Helper Functions**:
  - `calculate_depth(state)`: Calculates the depth of the state in the search tree.
  - `reconstruct_path(state)`: Reconstructs the path from the initial state to the current state.
  - `is_valid(state)`: Checks if all robots are on the end side.
  - `apply_action(state, action)`: Applies an action to a state to generate a new state.
  - `generate_actions(state)`: Generates possible actions (robot movements) from the current state.
  - `initialize(start_side_str, end_side_str)`: Initializes the starting state based on input.
  - `parse_input()`: Parses the input arguments for the program.

## Usage

1. **Running the Program**:
   To run the program, use the following command in the terminal:

        python bridge_crossing.py <start_side> <end_side> <algorithm>
  
	    •	<start_side>: A string representing the robots and power pack on the start side (e.g., “ACDP”).

	    •	<end_side>: A string representing the robots and power pack on the end side (e.g., “B”).

	    •	<algorithm>: The algorithm to use (UC for Uniform Cost Search or Astar for A* Search).

2.	Output:
The program will output the expanded nodes, solution cost, and the solution path if found.

Example
```bash
python bridge_crossing.py ABCDP "" Astar
```
OutPut:

    Running A* algorithm: 
    Expanded Nodes: {'A': True, 'B': True, 'C': True, 'D': True, 'P': True},
    Parent State: None,
    Time: 0, depth: 0
    ...
    Solution Path:
    Move A, B to the end side
    Move A back to the start side
    Move C, D to the end side
    Move B back to the start side
    Move A, B to the end side

Note

•	The program does not handle cases where the power pack is alone on one side.
•	Ensure all robots and the power pack are included in the input strings.

