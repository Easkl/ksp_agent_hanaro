import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 변수 정의
tracker_x = 0.0
tracker_y = 0.0
tracker_z = 0.0
vx_tracker = 0.0
vy_tracker = 0.0
vz_tracker = 0.0
target_x = 30.0
target_y = 30.0
target_z = 30.0
vx_target = 0.0
vy_target = 0.0
vz_target = 0.0

# error 정의, 상태벡터 : [error, rel_v]
error_x = tracker_x - target_x
error_y = tracker_y - target_y
error_z = tracker_z - target_z
rel_vx = vx_tracker - vx_target
rel_vy = vy_tracker - vy_target
rel_vz = vz_tracker - vz_target

# 상태벡터 정의
x = np.array([error_x, error_y, error_z, rel_vx, rel_vy, rel_vz])

T = 20.0
dt = 0.01
N = int(T / dt)

# 시스템 행렬 정의
A = np.array([[0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0., 1.],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.]])  # 상태: [ex, ey, ez, vx, vy, vz]
B = np.array([[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.],
              [1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]])          # 입력: [ux, uy, uz]
# 비용 행렬 정의
Q = np.array([[1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0.],
              [0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0., 1.]])  # 상태 비용 (나중에 가중치 조정 가능)
R = np.array([[1., 0., 0.], 
              [0., 1., 0.],
              [0., 0., 1.]])          # 입력 비용 (나중에 가중치 조정 가능)
# Riccati 방정식 풀기
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P  # K: 3x6 제어 이득 행렬

# 시뮬레이션 데이터 저장
tracker_x_hist = np.zeros(N)
tracker_y_hist = np.zeros(N)
tracker_z_hist = np.zeros(N)
target_x_hist = np.zeros(N)
target_y_hist = np.zeros(N)
target_z_hist = np.zeros(N)
tracker_x_hist[0] = tracker_x
tracker_y_hist[0] = tracker_y
tracker_z_hist[0] = tracker_z
target_x_hist[0] = target_x
target_y_hist[0] = target_y
target_z_hist[0] = target_z
u_hist = np.zeros((N, 3))
error_hist = np.zeros((N, 3))

for i in range(N - 1):
    # 목표 가속도: -50 ~ 50 랜덤
    ax_target = np.random.uniform(-500, 500)
    ay_target = np.random.uniform(-500, 500)
    az_target = np.random.uniform(-500, 500)

    # 목표 dynamics
    vx_target += ax_target * dt
    vy_target += ay_target * dt
    vz_target += az_target * dt
    target_x += vx_target * dt
    target_y += vy_target * dt
    target_z += vz_target * dt
    rel_a = np.array([ax_target, ay_target, az_target])

    target_x_hist[i+1] = target_x
    target_y_hist[i+1] = target_y
    target_z_hist[i+1] = target_z

    # 상대 상태 업데이트
    error_x = tracker_x - target_x
    error_y = tracker_y - target_y
    error_z = tracker_z - target_z
    rel_vx = vx_tracker - vx_target
    rel_vy = vy_tracker - vy_target
    rel_vz = vz_tracker - vz_target

    x = np.array([error_x, error_y, error_z, rel_vx, rel_vy, rel_vz])

    error_hist[i] = [error_x, error_y, error_z]

    # 제어 입력 계산
    u = -K @ x + rel_a # u: [ux, uy, uz]
    u_hist[i] = u

    # 추적자 dynamics 업데이트
    vx_tracker += u[0] * dt
    vy_tracker += u[1] * dt
    vz_tracker += u[2] * dt
    tracker_x += vx_tracker * dt
    tracker_y += vy_tracker * dt
    tracker_z += vz_tracker * dt

    tracker_x_hist[i+1] = tracker_x
    tracker_y_hist[i+1] = tracker_y
    tracker_z_hist[i+1] = tracker_z

# 마지막 샘플 기록
error_hist[-1] = [error_x, error_y, error_z]
u_hist[-1] = u

# =====================
# 3D 애니메이션
# =====================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D LQR Chaser-Target Animation")
ax.grid(True)

target_point, = ax.plot([], [], [], 'ro', markersize=10, label="Target")
tracker_point, = ax.plot([], [], [], 'bo', markersize=8, label="Tracker")
target_trail, = ax.plot([], [], [], 'r-', linewidth=1, alpha=0.7)
tracker_trail, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.7)

ax.legend()

def update(frame):
    # 목표 궤적
    target_trail.set_data(target_x_hist[:frame+1], target_y_hist[:frame+1])
    target_trail.set_3d_properties(target_z_hist[:frame+1])
    target_point.set_data([target_x_hist[frame]], [target_y_hist[frame]])
    target_point.set_3d_properties([target_z_hist[frame]])
    
    # 추적자 궤적
    tracker_trail.set_data(tracker_x_hist[:frame+1], tracker_y_hist[:frame+1])
    tracker_trail.set_3d_properties(tracker_z_hist[:frame+1])
    tracker_point.set_data([tracker_x_hist[frame]], [tracker_y_hist[frame]])
    tracker_point.set_3d_properties([tracker_z_hist[frame]])
    
    return target_point, tracker_point, target_trail, tracker_trail

ani = FuncAnimation(fig, update, frames=N, interval=50, blit=True)
plt.show()