import numpy as np
from control import tf, feedback, step_response, series
import control
from scipy.optimize import differential_evolution

# ==============================
# PyHesap Agent: Core Calculation and Simulation
# ==============================
class PyHesapAgent:
    def simulate_response(self, plant, pid_params, simulation_time=5, dt=0.01):
        Kp = pid_params['Kp']
        Ki = pid_params['Ki']
        Kd = pid_params['Kd']
        pid_controller = tf([Kd, Kp, Ki], [1, 0])
        open_loop = series(pid_controller, plant)
        closed_loop = feedback(open_loop, 1)
        t = np.arange(0, simulation_time, dt)
        t, y = step_response(closed_loop, t)
        settling_time = self.compute_settling_time(t, y)
        rise_time = self.compute_rise_time(t, y)
        overshoot = self.compute_overshoot(y)
        return settling_time, t, y, closed_loop, rise_time, overshoot

    def compute_settling_time(self, t, y, tolerance=0.02):
        final_value = y[-1]
        lower_bound = final_value * (1 - tolerance)
        upper_bound = final_value * (1 + tolerance)
        settling_time = t[-1]
        for i in range(len(y)):
            if all(lower_bound <= y[j] <= upper_bound for j in range(i, len(y))):
                settling_time = t[i]
                break
        return settling_time

    def compute_rise_time(self, t, y):
        final_value = y[-1]
        start_time = None
        end_time = None
        for i, val in enumerate(y):
            if start_time is None and val >= 0.1 * final_value:
                start_time = t[i]
            if start_time is not None and val >= 0.9 * final_value:
                end_time = t[i]
                break
        if start_time is not None and end_time is not None:
            return end_time - start_time
        else:
            return None

    def compute_overshoot(self, y):
        final_value = y[-1]
        max_val = np.max(y)
        if final_value == 0:
            return 0
        return max(0, (max_val - final_value) / abs(final_value) * 100)

# ==============================
# PSO Optimization (Core Algorithm)
# ==============================
def pso_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid,
                     swarm_size=20, max_iter=30):
    lb = np.array([0.1 * initial_pid['Kp'], 0.1 * initial_pid['Ki'], 0.1 * initial_pid['Kd']])
    ub = np.array([10 * initial_pid['Kp'], 10 * initial_pid['Ki'], 10 * initial_pid['Kd']])
    dim = 3
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    swarm_positions = np.random.uniform(lb, ub, (swarm_size, dim))
    swarm_velocities = np.random.uniform(-1, 1, (swarm_size, dim))
    pbest_positions = swarm_positions.copy()
    pbest_costs = np.full(swarm_size, np.inf)
    gbest_position = None
    gbest_cost = np.inf
    ph_agent = PyHesapAgent()

    def evaluate_cost(pid_vector):
        pid = {'Kp': pid_vector[0], 'Ki': pid_vector[1], 'Kd': pid_vector[2]}
        sim_time = max(5, 4 * target_settling_time)
        try:
            settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(plant, pid, simulation_time=sim_time)
        except Exception as ex:
            return 1e6
        cost_ts = abs(settling_time - target_settling_time)
        cost_os = max(0, overshoot - target_overshoot)
        cost_rt = max(0, rise_time - target_rise_time) if rise_time is not None else 1e3
        return cost_ts + 0.1 * cost_os + 0.1 * cost_rt

    for iter in range(max_iter):
        for i in range(swarm_size):
            cost = evaluate_cost(swarm_positions[i])
            if cost < pbest_costs[i]:
                pbest_costs[i] = cost
                pbest_positions[i] = swarm_positions[i].copy()
            if cost < gbest_cost:
                gbest_cost = cost
                gbest_position = swarm_positions[i].copy()
        for i in range(swarm_size):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            swarm_velocities[i] = (w * swarm_velocities[i] +
                                   c1 * r1 * (pbest_positions[i] - swarm_positions[i]) +
                                   c2 * r2 * (gbest_position - swarm_positions[i]))
            swarm_positions[i] = swarm_positions[i] + swarm_velocities[i]
            swarm_positions[i] = np.clip(swarm_positions[i], lb, ub)
    optimized_pid = {'Kp': gbest_position[0], 'Ki': gbest_position[1], 'Kd': gbest_position[2]}
    return optimized_pid

# ==============================
# Differential Evolution (DE) Optimization (Core Algorithm)
# ==============================
def de_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid):
    lb = [0.1 * initial_pid['Kp'], 0.1 * initial_pid['Ki'], 0.1 * initial_pid['Kd']]
    ub = [10 * initial_pid['Kp'], 10 * initial_pid['Ki'], 10 * initial_pid['Kd']]
    def cost_func(x):
        pid = {'Kp': x[0], 'Ki': x[1], 'Kd': x[2]}
        sim_time = max(5, 4 * target_settling_time)
        try:
            settling_time, t, y, closed_loop, rise_time, overshoot = PyHesapAgent().simulate_response(plant, pid, simulation_time=sim_time)
        except Exception as ex:
            return 1e6
        cost_ts = abs(settling_time - target_settling_time)
        cost_os = max(0, overshoot - target_overshoot)
        cost_rt = max(0, rise_time - target_rise_time) if rise_time is not None else 1e3
        return cost_ts + 0.1 * cost_os + 0.1 * cost_rt
    result = differential_evolution(cost_func, bounds=list(zip(lb, ub)), strategy='best1bin', maxiter=30, popsize=15)
    optimized_pid = {'Kp': result.x[0], 'Ki': result.x[1], 'Kd': result.x[2]}
    return optimized_pid

# ==============================
# Example Usage (Minimal)
# ==============================
if __name__ == "__main__":
    # Example plant: tf([1], [1, 2, 1])
    plant = tf([1], [1, 2, 1])
    targets = {'settling_time': 2.0, 'overshoot': 5.0, 'rise_time': 0.5}
    initial_pid = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01}

    # Optimize using PSO
    optimized_pid = pso_optimize_pid(plant, **targets, initial_pid=initial_pid)

    # Simulate
    ph_agent = PyHesapAgent()
    settling_time, t, y, _, rise_time, overshoot = ph_agent.simulate_response(plant, optimized_pid)

    print("Optimized PID:", optimized_pid)
    print("Settling Time:", settling_time)
    print("Overshoot:", overshoot)
    print("Rise Time:", rise_time)