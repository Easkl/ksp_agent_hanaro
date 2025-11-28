import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.linalg

time = 20.0
dt = 0.01
N = int(time / dt)
t = np.linspace(0, time, N)

# 시스템 행렬 (2D 상대계)
# 상태: [ex, ey, vx, vy] (ex = x_tracker - x_target, ey = y_tracker - y_target, vx = vx_tracker - vx_target, vy = vy_tracker - vy_target)
A = np.array([[0., 0., 1., 0.],
              [0., 0., 0., 1.],
              [0., 0., 0., 0.],
              [0., 0., 0., 0.]])  # dex/dt = vx, dey/dt = vy, dvx/dt = ux, dvy/dt = uy
B = np.array([[0., 0.],
              [0., 0.],
              [1., 0.],
              [0., 1.]])          # 입력 ux, uy

# 비용 행렬
Q = np.array([[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])  # 상태 비용
R = np.array([[1., 0.],
              [0., 1.]])          # 입력 비용

# Riccati 방정식 풀기
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P  # K: 2x4

# 초기 상태 (NumPy로 시뮬레이션)
x_tracker = 0.0
y_tracker = 0.0
vx_tracker = 0.0
vy_tracker = 0.0

x_target = 30.0
y_target = 0.0
vx_target = 0.0
vy_target = 0.0

# 시뮬레이션 데이터 저장
np.random.seed(42)
tracker_x = np.zeros(N)
tracker_y = np.zeros(N)
target_x = np.zeros(N)
target_y = np.zeros(N)

tracker_x[0] = x_tracker
tracker_y[0] = y_tracker
target_x[0] = x_target
target_y[0] = y_target

for i in range(N - 1):
    # 목표 가속도: -5 ~ 5 랜덤 (더 큰 범위로 움직임 증가)
    ax_target = np.random.uniform(-50, 50)
    ay_target = np.random.uniform(-50, 50)

    # 목표 dynamics
    vx_target += ax_target * dt
    vy_target += ay_target * dt
    x_target += vx_target * dt
    y_target += vy_target * dt

    target_x[i+1] = x_target
    target_y[i+1] = y_target

    # 상대 상태
    ex = x_tracker - x_target
    ey = y_tracker - y_target
    vx_rel = vx_tracker - vx_target
    vy_rel = vy_tracker - vy_target
    x_state = np.array([ex, ey, vx_rel, vy_rel])

    # 제어 입력
    u = -K @ x_state  # u: [ux, uy]

    # 추적자 dynamics
    vx_tracker += u[0] * dt
    vy_tracker += u[1] * dt
    x_tracker += vx_tracker * dt
    y_tracker += vy_tracker * dt

    tracker_x[i+1] = x_tracker
    tracker_y[i+1] = y_tracker

# 애니메이션
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("2D LQR Chaser-Target Animation")
ax.grid(True)
ax.set_aspect('equal')

target_point, = ax.plot([], [], 'ro', markersize=10, label="Target")
chaser_point, = ax.plot([], [], 'bo', markersize=10, label="Chaser")

ax.legend()

def init():
    target_point.set_data([], [])
    chaser_point.set_data([], [])
    return target_point, chaser_point

def animate(frame):
    target_point.set_data([target_x[frame]], [target_y[frame]])
    chaser_point.set_data([tracker_x[frame]], [tracker_y[frame]])
    return target_point, chaser_point

anim = FuncAnimation(fig, animate, init_func=init, frames=N, interval=10, blit=True, repeat=True)

plt.show()
