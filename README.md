# Deep Diverging Diamond Simulation

This project simulates a deep diverging diamond interchange and uses reinforcement learning to optimize traffic flow.

## Overview

The project consists of two main components:
1. **Interactive Simulation**: Allows users to interact with the simulation in real-time.
2. **Model Training**: Uses the DDPG (Deep Deterministic Policy Gradient) algorithm to train a model that optimizes the traffic flow in the simulation.

## Getting Started

### Interactive Simulation

To run the interactive simulation, call the `main.py` script. This will start the simulation and allow you to interact with it in real-time.

### Model Training

To train the model, run the `ddpg_model.py` script. This will start the training process using the DDPG algorithm.

### Tweaking the Model

The `model_variables.py` file contains various parameters and hyperparameters that can be adjusted to tweak the model's performance. These include learning rates, discount factors, and network architecture settings. By modifying these variables, you can experiment with different configurations to find the optimal settings for your simulation.

To adjust these parameters, open the `model_variables.py` file and modify the values as needed. For example, you can change the learning rate to see how it affects the training process. After making your changes, save the file and re-run the `ddpg_model.py` script to train the model with the new settings.