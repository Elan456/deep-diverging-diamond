import numpy as np
import tensorflow as tf
from collections import deque




lights_state = [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]


# Create the DQN model
model = tf.keras.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(len(lights),)), 
    # Hidden layers
    tf.keras.layers.Dense(64, activation='relu'),
    # ... more layers if needed
    # Output layer with Q-values for each action
    tf.keras.layers.Dense(num_actions)  
])

# Compile the model
model.compile(optimizer='adam', loss='mse')