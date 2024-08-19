from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


def get_vehicle_numbers(lanes):
    vehicle_per_lane = {lane: 0 for lane in lanes}
    for lane in lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        for vehicle_id in vehicle_ids:
            if traci.vehicle.getLanePosition(vehicle_id) > 10:
                vehicle_per_lane[lane] += 1
    return vehicle_per_lane


def get_waiting_time(lanes):
    return sum(traci.lane.getWaitingTime(lane) for lane in lanes)


def set_phase_duration(junction, duration, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, duration)


class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.lr = lr
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims, batch_size, n_actions, junctions, max_memory_size=100000, epsilon_dec=5e-4, epsilon_end=0.05, bias_weight=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.iter_cntr = 0
        self.replace_target = 100
        self.bias_weight = bias_weight

        self.Q_eval = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.memory = {junction: {'state_memory': np.zeros((max_memory_size, input_dims), dtype=np.float32),
                                  'new_state_memory': np.zeros((max_memory_size, input_dims), dtype=np.float32),
                                  'reward_memory': np.zeros(max_memory_size, dtype=np.float32),
                                  'action_memory': np.zeros(max_memory_size, dtype=np.int32),
                                  'terminal_memory': np.zeros(max_memory_size, dtype=bool),
                                  'mem_cntr': 0} for junction in junctions}

    def store_transition(self, state, new_state, action, reward, done, junction):
        mem_cntr = self.memory[junction]['mem_cntr'] % self.max_mem
        self.memory[junction]['state_memory'][mem_cntr] = state
        self.memory[junction]['new_state_memory'][mem_cntr] = new_state
        self.memory[junction]['reward_memory'][mem_cntr] = reward
        self.memory[junction]['terminal_memory'][mem_cntr] = done
        self.memory[junction]['action_memory'][mem_cntr] = action
        self.memory[junction]['mem_cntr'] += 1

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def learn(self, junction):
        if self.memory[junction]['mem_cntr'] < self.batch_size:
            return

        max_mem = min(self.memory[junction]['mem_cntr'], self.max_mem)
        batch_indices = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.memory[junction]['state_memory'][batch_indices]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]['new_state_memory'][batch_indices]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch_indices]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch_indices]).to(self.Q_eval.device)
        action_batch = self.memory[junction]['action_memory'][batch_indices]

        q_eval = self.Q_eval(state_batch)[np.arange(self.batch_size), action_batch]
        q_next = self.Q_eval(new_state_batch).max(dim=1)[0]
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next

        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.epsilon - self.epsilon_dec, self.epsilon_end)
        self.iter_cntr += 1


def run_simulation(train=True, model_name="model", epochs=50, steps=500, arduino=False):
    if arduino:
        arduino_conn = serial.Serial(port="COM4", baudrate=9600, timeout=0.1)

        def write_read(x):
            arduino_conn.write(bytes(x, "utf-8"))
            time.sleep(0.05)
            return arduino_conn.readline()

    best_time = np.inf
    total_time_list = []

    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])
    junctions = traci.trafficlight.getIDList()
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=4, fc1_dims=256, fc2_dims=256, batch_size=64, n_actions=4, junctions=junctions)

    if not train:
        agent.Q_eval.load_state_dict(torch.load(f"models/{model_name}.bin", map_location=agent.Q_eval.device))

    traci.close()
    for epoch in range(epochs):
        if train:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", "--tripinfo-output", "tripinfo.xml"])

        step = 0
        total_waiting_time = 0
        min_duration = 5

        for junction in junctions:
            prev_vehicles = {junction: [0] * 4}
            traffic_light_times = {junction: 0}

        while step <= steps:
            traci.simulationStep()

            for junction in junctions:
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controlled_lanes)
                total_waiting_time += waiting_time

                if traffic_light_times[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controlled_lanes)
                    state = prev_vehicles[junction]
                    new_state = list(vehicles_per_lane.values())
                    action = agent.choose_action(new_state)
                    next_state = new_state.copy()
                    next_state[action] += 1

                    reward = -waiting_time
                    agent.store_transition(state, next_state, action, reward, step == steps, junction)

                    phase = action % 4
                    set_phase_duration(junction, 6, ["G" * 4 + "r" * 12, "r" * 16][phase])
                    set_phase_duration(junction, min_duration + 10, ["r" * 12 + "G" * 4, "r" * 16][phase])

                    if arduino:
                        write_read(str(traci.trafficlight.getPhase(junction)))

                    traffic_light_times[junction] = min_duration + 10
                    if train:
                        agent.learn(junction)
                else:
                    traffic_light_times[junction] -= 1

            step += 1

        total_time_list.append(total_waiting_time)

        if total_waiting_time < best_time:
            best_time = total_waiting_time
            if train:
                torch.save(agent.Q_eval.state_dict(), f"models/{model_name}.bin")

        traci.close()

    if train:
        plt.plot(range(len(total_time_list)), total_time_list)
        plt.xlabel("Epochs")
        plt.ylabel("Total Waiting Time")
        plt.savefig(f"models/{model_name}.png")
        plt.show()

    return agent


def main():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--train", action="store_true", default=False, help="Train the model")
    opt_parser.add_option("--epochs", type="int", default=50, help="Number of epochs to run the simulation")
    opt_parser.add_option("--steps", type="int", default=500, help="Number of steps per epoch")
    opt_parser.add_option("--model_name", type="str", default="model", help="Model name for saving/loading")
    opt_parser.add_option("--arduino", action="store_true", default=False, help="Enable Arduino communication")
    options, _ = opt_parser.parse_args()

    run_simulation(train=options.train, model_name=options.model_name, epochs=options.epochs, steps=options.steps, arduino=options.arduino)


if __name__ == "__main__":
    main()
