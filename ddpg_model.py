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
        x = self.relu(self.fc3(x))
        # Scale output to [0, 1]
        x = (x + 1) / 2
        return x


def compute_policy_gradient(states, actions, reward, rewards):
  policy_gradient = []
  gamma = 0.99
  last_tick = states[-1].current_tick
  for state, action, instant_reward in zip(states, actions, rewards):
      current_tick = state.current_tick
      state = torch.tensor(state.induction_plate_last_activated + state.current_induction_plate_states, dtype=torch.float)
      action = torch.tensor(action, dtype=torch.long)
      log_prob = torch.log(actor(state)[action]  + 1e-8)
      grad = -log_prob * (reward * (gamma ** (last_tick - current_tick)) + instant_reward)
      policy_gradient.append(grad)
      
  print("Gradients", policy_gradient)
  return policy_gradient


def simulation_episode(actor, sim, cars=None, inputFile=None, episode_number=None, to_print=False):
  if cars:
    sim.set_scenario(cars=cars)
  elif inputFile:
    sim.set_scenario(inputFile=inputFile)
  else:
    raise ValueError("Must provide either cars or inputFile")
  
  states= []
  actions = []
  rewards = []
  state = sim.get_state()
  states.append(state)
  done = state.all_cars_done
  count = 0
  crash_count = 0
  while not done and count < 300:
    # print(state.current_induction_plate_states + state.induction_plate_last_activated)
    action_probs = actor(torch.tensor(state.induction_plate_last_activated + state.current_induction_plate_states, dtype=torch.float))
    action_probs = np.array(action_probs.detach().numpy())
    # print("Actions probs", action_probs)
    selected_action = [1 if prob > 0.5 else 0 for prob in action_probs]
    actions.append(action_probs)
    sim.apply_action(selected_action)
    if episode_number is not None:
      sim.step(episode_number)
    else:
      sim.step()
    state = sim.get_state()
    states.append(state)
    rewards.append(0)
    if state.crash_occurred:
      # print("Crash occurred")
      crash_count += 1
      rewards[-8:-2] = [r - 300 for r in rewards[-8:-2]]
    count += 1
    done = state.all_cars_done
  average_time = sim.get_average_time()
  if to_print:
    print(f"Average time for cars: {average_time} and crashes: {crash_count}")
  # Create a reward that is the negative of the average time
  reward = -average_time  
  return states, actions, reward, rewards

def train(actor, sim, cars=None, episode_count=10):
  total_reward = 0
  actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
  to_print = False
  for i in range(episode_count):

    to_print = True if i % 10 == 0 else False
    states, actions, reward, rewards = simulation_episode(actor, sim, cars=cars, episode_number=i, to_print=to_print)
    # print(f"Episode {i+1} Reward: {reward}")
    total_reward += reward
    policy_grad = compute_policy_gradient(states, actions, reward, rewards)
    actor_optimizer.zero_grad()
    for param in actor.parameters():
      param.grad = torch.zeros_like(param)
    for grad in policy_grad:
      for param, g in zip(actor.parameters(), grad):
        param.grad += g
    actor_optimizer.step()

  # print(f"Average reward: {total_reward/10}")

def create_and_save_actor(state_dim, action_dim, hidden_dim, file_path):
  actor = ActorNetwork(state_dim, action_dim, hidden_dim)
  torch.save(actor.state_dict(), file_path)
  print(f"Actor model saved to {file_path}")

def load_actor_train(file_path, state_dim, action_dim, hidden_dim):
  actor = ActorNetwork(state_dim, action_dim, hidden_dim)
  actor.load_state_dict(torch.load(file_path, weights_only=True))
  actor.train()
  return actor

def load_actor_eval(file_path, state_dim, action_dim, hidden_dim):
  actor = ActorNetwork(state_dim, action_dim, hidden_dim)
  actor.load_state_dict(torch.load(file_path, weights_only=True))
  actor.eval()
  return actor

def update_actor_file(actor, file_path):
  torch.save(actor.state_dict(), file_path)

def train_with_cars(actor, sim, car_count, episode_count):
  cars = []
  for _ in range(car_count):
    cars.append((random.randint(0, 15), random.randint(0, 100)))
  train(actor, sim, cars, episode_count)

def train_one_epoch(actor, sim, car_count, episode_count):
  train_with_cars(actor, sim, car_count, episode_count)

if __name__ == "__main__":
  sim = Simulation(render=True, render_frequency=10)
  
  # Initialize networks and optimizers
  state_dim = 24
  action_dim = 12
  hidden_dim = 24
  espisode_count = 10
  # create_and_save_actor(state_dim, action_dim, hidden_dim, "actor1.pth")
  car_count = 50 
  cars = []
  for _ in range(car_count):
      cars.append((random.randint(0, 15), random.randint(0, 100)))
  actor = load_actor_train("actor1.pth", state_dim, action_dim, hidden_dim)

  for i in range(30):
    train_one_epoch(actor, sim, car_count, espisode_count)
    
  # print([param for param in actor.parameters()])
  sim = Simulation(render=True, render_frequency=1)
  
  simulation_episode(actor, sim, cars=cars, episode_number=1, to_print=True)  
  






