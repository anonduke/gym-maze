# CSCN8020 Assignment 1: Reinforcement Learning Programming

## Overview
Assignment1 contains the complete solution for Assignment 1 of the CSCN8020 course on Reinforcement Learning Programming (assignment.ipynb). The assignment covers designing an MDP for a pick-and-place robot, manual value iteration on a 2x2 gridworld, implementing value iteration variants on a 5x5 gridworld, and off-policy Monte Carlo with importance sampling on the same 5x5 gridworld.

# CSCN8020 Assignment 2: Reinforcement Learning Programming
The Taxi environment is a grid-based problem where an agent (a taxi) must navigate to pick up a passenger and deliver them to a designated location. The environment features:

6 discrete actions: Move south, move north, move east, move west, pickup passenger, drop off passenger.
500 discrete states: Determined by 25 taxi positions, 5 passenger locations (including in the taxi), and 4 destination locations (Red, Green, Yellow, Blue).
Rewards:

+20 for delivering a passenger.
-10 for illegal pickup or drop-off actions.
-1 per step otherwise.


Project Structure

- assignment2.ipynb: Jupyter notebook containing the Q-Learning implementation, training, and analysis.
- assignment2_utils.py: Utility file with helper functions for environment interaction and observation description.

## Requirements
- Python 3.8+
- Install requirements.txt 




