import numpy as np
from control import tf, feedback, step_response, series, margin
import control
import warnings
import openai
import os
import json
from scipy.optimize import differential_evolution

# Suppress warning messages.
warnings.filterwarnings("ignore")

# ==============================
# OpenAI Settings for NLP
# ==============================

import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.default_headers = {"x-foo": "true"}

# -----------------------------
# ALGORITHM REGISTRY
# -----------------------------
ALGORITHM_REGISTRY = {}

def register_algorithm(name):
    """Decorator to register a new optimization algorithm."""
    def decorator(fn):
        ALGORITHM_REGISTRY[name] = fn
        return fn
    return decorator

# ==============================
# Helper Function: Convert Transfer Function to LaTeX Format
# ==============================
def tf_to_latex(num, den):
    def poly_to_string(coeffs):
        terms = []
        degree = len(coeffs) - 1
        for i, coef in enumerate(coeffs):
            power = degree - i
            if abs(coef) < 1e-12:
                continue
            if power == 0:
                term = f"{coef:g}"
            elif power == 1:
                term = f"{coef:g}s" if abs(coef-1) >= 1e-12 else "s"
            else:
                term = f"{coef:g}s^{power}" if abs(coef-1) >= 1e-12 else f"s^{power}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"
    num_str = poly_to_string(num)
    den_str = poly_to_string(den)
    return r"$\frac{" + num_str + "}{" + den_str + "}$"

# ==============================
# Helper Function: Render Formula Image (Replaced with text)
# ==============================
def render_formula_image(latex_formula):
    # Instead of rendering image, return the LaTeX string as text
    return latex_formula

# ==============================
# Helper Function: Compute Z–N Ultimate Gain and Period
# ==============================
def compute_zn_params(plant):
    # 1) Log–space frequency sweep
    ws = np.logspace(-2, 2, 5000)
    mag, phase, _ = control.freqresp(plant, ws)
    # Convert phase to degrees and unwrap
    phase_deg = np.unwrap(np.angle(phase, deg=True))

    # 2) Find index where phase crosses or is closest to –180°
    idxs = np.where(phase_deg <= -180)[0]
    if len(idxs) == 0:
        idx = np.argmin(np.abs(phase_deg + 180))
    else:
        idx = idxs[0]

    ωu = ws[idx]
    Ku = 1.0 / abs(mag[idx])
    Tu = 2 * np.pi / ωu
    return Ku, Tu

# ==============================
# Functions: Extract Target Values from Problem Description
# ==============================
def extract_target_settling_time(description):
    prompt = (
        "Extract the target settling time in seconds from the following control problem description. "
        "If no target is specified or the value is unclear, output 0.3. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 0.3
    except Exception as e:
        print("Error extracting settling time:", e)
        return 0.3

def extract_target_overshoot(description):
    prompt = (
        "Extract the target maximum overshoot percentage from the following control problem description. "
        "If not specified, output 5. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 5.0
    except Exception as e:
        print("Error extracting overshoot:", e)
        return 5.0

def extract_target_rise_time(description):
    prompt = (
        "Extract the target minimum rise time in seconds from the following control problem description. "
        "If not specified, output 0.5. Only output the number.\n\n"
        f"Description: \"{description}\""
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Extract a number from the text and answer in English."},
                      {"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        try:
            return float(result)
        except:
            return 0.5
    except Exception as e:
        print("Error extracting rise time:", e)
        return 0.5

# ==============================
# Sample Transfer Functions (20 samples, original versions)
# ==============================
sample_transfer_functions = [
    {"name": "Default Transfer Function", "num": [1], "den": [1,2,1], "desc": "Default system: tf([1], [1,2,1])"},
    {"name": "Example 1: Second-Order System", "num": [1], "den": [1,2,1], "desc": "Standard second-order system."},
    {"name": "Example 2: First-Order System", "num": [1], "den": [1,1], "desc": "First-order system."},
    {"name": "Example 3: Delayed System", "num": [1], "den": [1,3,2], "desc": "System with delay dynamics."},
    {"name": "Example 4: Low Damping System", "num": [1], "den": [1,0.5,1], "desc": "Second-order system with low damping."},
    {"name": "Example 5: High-Speed System", "num": [10], "den": [1,10], "desc": "Fast responding system."},
    {"name": "Example 6: Slow System", "num": [0.5], "den": [1,0.1], "desc": "Slow dynamic system."},
    {"name": "Example 7: Second Order with Low Damping", "num": [1], "den": [1,0.2,1], "desc": "Low damping system."},
    {"name": "Example 8: Second Order with High Damping", "num": [1], "den": [1,5,1], "desc": "High damping system."},
    {"name": "Example 9: Third-Order System", "num": [1], "den": [1,3,3,1], "desc": "Third-order system."},
    {"name": "Example 10: Simple Integrator", "num": [1], "den": [1,0], "desc": "Integrator system."},
    {"name": "Example 11: Simple Differentiator", "num": [1,0], "den": [1], "desc": "Differentiator system."},
    {"name": "Example 12: PID Example", "num": [0.01,1,0.5], "den": [1,0], "desc": "PID control structure (example values)."},
    {"name": "Example 13: Triple Polynomial", "num": [1], "den": [1,1,1], "desc": "Simple triple polynomial system."},
    {"name": "Example 14: Special Second-Order", "num": [2], "den": [1,4,4,1], "desc": "Special form second-order system."},
    {"name": "Example 15: Scaled System", "num": [1], "den": [1,2.5,1.5], "desc": "System with varied coefficients."},
    {"name": "Example 16: Third Order Polynomial", "num": [1], "den": [1,6,11,6], "desc": "Typical third-order system."},
    {"name": "Example 17: Minimal Damping", "num": [1], "den": [1,0.8,0.16], "desc": "System with very low damping."},
    {"name": "Example 18: Fourth-Order System", "num": [1], "den": [1,3,3,1,0.5], "desc": "A fourth-order system."},
    {"name": "Example 19: Simple Multi-Order System", "num": [0.5], "den": [1,2,2,1], "desc": "Balanced multi-order system."},
    {"name": "Example 20: Flat System", "num": [1], "den": [1,1,1,1], "desc": "Simple flat system."}
]

# ==============================
# PSO Optimization
# ==============================
@register_algorithm("PSO")
def pso_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid,
                     swarm_size=20, max_iter=30):
    lb = np.array([2 * initial_pid['Kp'], 2 * initial_pid['Ki'], 1 * initial_pid['Kd']])
    ub = np.array([300 * initial_pid['Kp'], 200 * initial_pid['Ki'], 100 * initial_pid['Kd']])
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
        reg_term = 0.001 * (pid['Kp']**2 + pid['Ki']**2 + pid['Kd']**2)
        return 1.0 * cost_ts + 0.1 * cost_os + 0.1 * cost_rt + reg_term

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
        print(f"PSO Iteration {iter+1}/{max_iter}, Best Cost: {gbest_cost:.4f}")
    optimized_pid = {'Kp': gbest_position[0], 'Ki': gbest_position[1], 'Kd': gbest_position[2]}
    return optimized_pid

# ==============================
# Differential Evolution (DE) Optimization
# ==============================
@register_algorithm("DE")
def de_optimize_pid(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid):
    lb = [0.1 * initial_pid['Kp'], 0.1 * initial_pid['Ki'], 0.1 * initial_pid['Kd']]
    ub = [100 * initial_pid['Kp'], 100 * initial_pid['Ki'], 100 * initial_pid['Kd']]
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
        reg_term = 0.001 * (pid['Kp']**2 + pid['Ki']**2 + pid['Kd']**2)
        return 1.0 * cost_ts + 0.1 * cost_os + 0.1 * cost_rt + reg_term
    result = differential_evolution(cost_func, bounds=list(zip(lb, ub)), strategy='best1bin', maxiter=30, popsize=15)
    optimized_pid = {'Kp': result.x[0], 'Ki': result.x[1], 'Kd': result.x[2]}
    return optimized_pid

# ==============================
# PyHesap Agent: Calculation, Simulation, and Metric Computation
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
# Main CLI Application
# ==============================
if __name__ == "__main__":
    print("SmartControl PID Design (CLI Version)")
    print("Enter control problem description:")
    description = input("> ")

    # Extract targets
    target_settling_time = extract_target_settling_time(description)
    target_overshoot = extract_target_overshoot(description)
    target_rise_time = extract_target_rise_time(description)

    print(f"Target Settling Time: {target_settling_time} s")
    print(f"Target Overshoot: {target_overshoot}%")
    print(f"Target Rise Time: {target_rise_time} s")

    # Input transfer function
    print("Enter transfer function numerator (comma-separated):")
    num_str = input("> ")
    print("Enter transfer function denominator (comma-separated):")
    den_str = input("> ")

    try:
        num = [float(x) for x in num_str.split(',')]
        den = [float(x) for x in den_str.split(',')]
        plant = tf(num, den)
    except Exception as e:
        print(f"Error creating transfer function: {e}")
        exit(1)

    # Choose algorithm
    print("Choose optimization algorithm (PSO or DE):")
    alg = input("> ").upper()
    if alg not in ALGORITHM_REGISTRY:
        print("Invalid algorithm. Using PSO.")
        alg = "PSO"

    # Initial PID
    initial_pid = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.01}

    # Optimize
    optimize_func = ALGORITHM_REGISTRY[alg]
    optimized_pid = optimize_func(plant, target_settling_time, target_overshoot, target_rise_time, initial_pid)

    # Simulate
    ph_agent = PyHesapAgent()
    sim_time = max(5, 4 * target_settling_time)
    settling_time, t, y, closed_loop, rise_time, overshoot = ph_agent.simulate_response(plant, optimized_pid, simulation_time=sim_time)

    # Output results
    print("\nOptimized PID:")
    print(f"Kp: {optimized_pid['Kp']:.4f}")
    print(f"Ki: {optimized_pid['Ki']:.4f}")
    print(f"Kd: {optimized_pid['Kd']:.4f}")
    print(f"\nFinal Settling Time: {settling_time:.3f} s (Target: {target_settling_time} s)")
    print(f"Final Overshoot: {overshoot:.2f}% (Target: {target_overshoot}%)")
    print(f"Final Rise Time: {rise_time:.3f} s (Target: {target_rise_time} s)")

    # Save to JSON
    result = {
        "optimized_pid": optimized_pid,
        "targets": {"settling_time": target_settling_time, "overshoot": target_overshoot, "rise_time": target_rise_time},
        "final_metrics": {"settling_time": settling_time, "overshoot": overshoot, "rise_time": rise_time}
    }
    with open("pid_result.json", "w") as f:
        json.dump(result, f, indent=4)
    print("\nResults saved to pid_result.json")