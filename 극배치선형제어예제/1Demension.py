import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
dt = 0.01
T = 20.0
N = int(T / dt)

t = np.linspace(0, T, N)

# Set random seed for reproducibility
np.random.seed(42)

# Allocate arrays
x_t = np.zeros(N)      # target position
v_t = np.zeros(N)      # target velocity

x_c = np.zeros(N)      # chaser position
v_c = np.zeros(N)      # chaser velocity

u = np.zeros(N)        # control input

# Initial conditions
x_t[0] = 0.0
v_t[0] = 0.0

x_c[0] = -20.0  # start further behind target
v_c[0] = 0.0

# Controller gain (pole placement result: K = [6, 5])
k1, k2 = 6.0, 5.0

# Simulation loop
for i in range(N - 1):
    # Target dynamics with random acceleration (much faster)
    a_t = 10 * np.random.normal(0, 1.0)  # Adjusted scale for visible fast movement
    v_t[i+1] = v_t[i] + a_t * dt
    x_t[i+1] = x_t[i] + v_t[i] * dt

    # Relative states
    e = x_c[i] - x_t[i]
    v_rel = v_c[i] - v_t[i]

    # Control law: u = a_t(t) - K z, z = [e, v_rel]
    u[i] = a_t - (k1 * e + k2 * v_rel)

    # Chaser dynamics
    v_c[i+1] = v_c[i] + u[i] * dt
    x_c[i+1] = x_c[i] + v_c[i] * dt

# Last control input (for plotting consistency)
u[-1] = u[-2]

# Animation setup
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(-100, 100)  # Fixed wider range for faster movement
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Position")
ax.set_title("1D Chaser-Target Animation")
ax.grid(True)

# Plot the line
ax.axhline(y=0, color='black', linewidth=1)

# Points for target and chaser
target_point, = ax.plot([], [], 'ro', markersize=10, label="Target")
chaser_point, = ax.plot([], [], 'bo', markersize=10, label="Chaser")

ax.legend()

def init():
    target_point.set_data([], [])
    chaser_point.set_data([], [])
    return target_point, chaser_point

def animate(frame):
    target_point.set_data([x_t[frame]], [0])
    chaser_point.set_data([x_c[frame]], [0])
    return target_point, chaser_point

anim = FuncAnimation(fig, animate, init_func=init, frames=N, interval=1, blit=True, repeat=True)

plt.show()
