import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
dt = 0.01
T = 20.0
N = int(T / dt)

t = np.linspace(0, T, N)

# Set random seed for reproducibility
np.random.seed(42)

# Allocate arrays for 3D
x_t = np.zeros(N)      # target x position
y_t = np.zeros(N)      # target y position
z_t = np.zeros(N)      # target z position
vx_t = np.zeros(N)     # target x velocity
vy_t = np.zeros(N)     # target y velocity
vz_t = np.zeros(N)     # target z velocity

x_c = np.zeros(N)      # chaser x position
y_c = np.zeros(N)      # chaser y position
z_c = np.zeros(N)      # chaser z position
vx_c = np.zeros(N)     # chaser x velocity
vy_c = np.zeros(N)     # chaser y velocity
vz_c = np.zeros(N)     # chaser z velocity

ux = np.zeros(N)       # control input x
uy = np.zeros(N)       # control input y
uz = np.zeros(N)       # control input z

# Initial conditions
x_t[0] = 0.0
y_t[0] = 0.0
z_t[0] = 0.0
vx_t[0] = 0.0
vy_t[0] = 0.0
vz_t[0] = 0.0

x_c[0] = -20.0  # start further behind target
y_c[0] = 0.0
z_c[0] = 0.0
vx_c[0] = 0.0
vy_c[0] = 0.0
vz_c[0] = 0.0

# Controller gain (pole placement result: K = [10, 5, 10, 5, 10, 5] for [dx, dvx, dy, dvy, dz, dvz])
kx1, kx2, ky1, ky2, kz1, kz2 = 10.0, 5.0, 10.0, 5.0, 10.0, 5.0

# Simulation loop
for i in range(N - 1):
    # Target dynamics with random acceleration (much faster)
    ax_t = 200 * np.random.normal(0, 1.0)
    ay_t = 200 * np.random.normal(0, 1.0)
    az_t = 200 * np.random.normal(0, 1.0)
    vx_t[i+1] = vx_t[i] + ax_t * dt
    vy_t[i+1] = vy_t[i] + ay_t * dt
    vz_t[i+1] = vz_t[i] + az_t * dt
    x_t[i+1] = x_t[i] + vx_t[i] * dt
    y_t[i+1] = y_t[i] + vy_t[i] * dt
    z_t[i+1] = z_t[i] + vz_t[i] * dt

    # Relative states
    dx = x_c[i] - x_t[i]
    dy = y_c[i] - y_t[i]
    dz = z_c[i] - z_t[i]
    dvx = vx_c[i] - vx_t[i]
    dvy = vy_c[i] - vy_t[i]
    dvz = vz_c[i] - vz_t[i]

    # Control law
    ux[i] = ax_t - (kx1 * dx + kx2 * dvx)
    uy[i] = ay_t - (ky1 * dy + ky2 * dvy)
    uz[i] = az_t - (kz1 * dz + kz2 * dvz)

    # Chaser dynamics
    vx_c[i+1] = vx_c[i] + ux[i] * dt
    vy_c[i+1] = vy_c[i] + uy[i] * dt
    vz_c[i+1] = vz_c[i] + uz[i] * dt
    x_c[i+1] = x_c[i] + vx_c[i] * dt
    y_c[i+1] = y_c[i] + vy_c[i] * dt
    z_c[i+1] = z_c[i] + vz_c[i] * dt

# Last control inputs
ux[-1] = ux[-2]
uy[-1] = uy[-2]
uz[-1] = uz[-2]

# Animation setup
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_zlim(-200, 200)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Chaser-Target Animation")

# Points for target and chaser
target_point, = ax.plot([], [], [], 'ro', markersize=10, label="Target")
chaser_point, = ax.plot([], [], [], 'bo', markersize=10, label="Chaser")

ax.legend()

def init():
    target_point.set_data([], [])
    target_point.set_3d_properties([])
    chaser_point.set_data([], [])
    chaser_point.set_3d_properties([])
    return target_point, chaser_point

def animate(frame):
    target_point.set_data([x_t[frame]], [y_t[frame]])
    target_point.set_3d_properties([z_t[frame]])
    chaser_point.set_data([x_c[frame]], [y_c[frame]])
    chaser_point.set_3d_properties([z_c[frame]])
    return target_point, chaser_point

anim = FuncAnimation(fig, animate, init_func=init, frames=N, interval=1, blit=False, repeat=True)

plt.show()
