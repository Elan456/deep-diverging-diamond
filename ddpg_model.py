import random
import numpy as np
from simulation.simulation import Simulation
import os
import torch
import torch.nn as nn
import torch.optim as optim


# Define the actor network
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return (x + 1) / 2  # Scale output to [0, 1]

# Define the critic network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def __len__(self):
        return len(self.buffer)

gamma = 0.99  # Discount factor

sim = Simulation()

def train_ddpg_episode(actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, states, actions, reward, gamma=0.99, tau=0.005):
  
  total_reward = torch.tensor([reward], dtype=torch.float)

  for state, action in zip(states, actions):
    state = torch.tensor(state.induction_plate_last_activated, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.float)
    state = state.unsqueeze(0) if state.dim() == 1 else state
    action = action.unsqueeze(0) if action.dim() == 1 else action


    #critic update
    q_value = critic(state, action)
    critic_loss = nn.MSELoss()(q_value, total_reward.unsqueeze(0))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    #actor update
    actor_loss = -critic(state, actor(state)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


  # Update the target networks
  for param, target_param in zip(actor.parameters(), target_actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

  for param, target_param in zip(critic.parameters(), target_critic.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  return actor, critic, target_actor, target_critic
def simulation_episode(actor, cars=None, inputFile=None):
  sim = Simulation()
  if cars:
    sim.set_scenario(cars=cars)
  elif inputFile:
    sim.set_scenario(inputFile=inputFile)
  else:
    raise ValueError("Must provide either cars or inputFile")
  
  states= []
  actions = []
  state = sim.get_state()
  states.append(state)
  done = state.all_cars_done
  count = 0
  while not done and count < 1000:
    action = actor(torch.tensor(state.induction_plate_last_activated, dtype=torch.float))
    actions.append(action)
    sim.apply_action(action)
    sim.step()
    state = sim.get_state()
    states.append(state)
    count += 1
    done = state.all_cars_done
  average_time = sim.get_average_time()
  # Take the inverse of the reward
  print(f"Average time for cars: {average_time} for episode with {len(states)} steps")
  # Create a reward that is the inverse of the average time
  reward = 1 / average_time  
  return states, actions, reward

# List all files in the ./sims folder
sims_folder = 'simulation/sim_defs'
sim_files = [os.path.join(sims_folder, f) for f in os.listdir(sims_folder) if os.path.isfile(os.path.join(sims_folder, f)) and f.endswith('.csv')]

# Initialize networks and optimizers
state_dim = 12
action_dim = 12
hidden_dim = 128
# Define the actor networks
actor = ActorNetwork(state_dim, action_dim, hidden_dim)
target_actor = ActorNetwork(state_dim, action_dim, hidden_dim)
target_actor.load_state_dict(actor.state_dict())

# Define the critic networks
critic = CriticNetwork(state_dim, action_dim, hidden_dim)
target_critic = CriticNetwork(state_dim, action_dim, hidden_dim)
target_critic.load_state_dict(critic.state_dict())

# Define the optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
for i in range(100):
  cars = []
  for _ in range(100):
      cars.append((random.randint(0, 15), random.randint(0, 100)))
  states, actions, reward = simulation_episode(actor, cars=cars, inputFile=None)

  actor, critic, target_actor, target_critic = train_ddpg_episode(
    actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, states, actions, reward
  )  # Example number of episodes

