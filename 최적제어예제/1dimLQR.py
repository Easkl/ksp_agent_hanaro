import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.linalg

time = 20.0
dt = 0.01
N = int(time / dt)
t = np.linspace(0, time, N)

# 시스템 행렬
# 상대계에서: e = x_tracker - x_target, v = v_tracker - v_target
# de/dt = v, dv/dt = u  라고 보는 선형 모델
A = np.array([[0., 1.],
              [0., 0.]])   # dx/dt = v, dv/dt = u (상대계에서 제어입력만 본 선형 모델)
B = np.array([[0.],
              [1.]])       # 입력이 가속도에 영향

# 비용 행렬
Q = np.array([[1., 0.],
              [0., 1.]])   # 상태 비용
R = np.array([[1.]])       # 입력 비용

# Riccati 방정식 풀기
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P  # K 계산

# 초기 상태 (NumPy로 시뮬레이션)
x_tracker = 0.0
x_tracker_dot = 0.0
x_target = 5.0     # 가까운 목표
x_target_dot = 0.0 # 목표 속도

# 시뮬레이션 데이터 저장
np.random.seed(42)  # 재현 가능하게
tracker_pos = np.zeros(N)
target_pos  = np.zeros(N)
u_hist      = np.zeros(N)
e_hist      = np.zeros(N)
v_rel_hist  = np.zeros(N)

tracker_pos[0] = x_tracker
target_pos[0]  = x_target

for i in range(N - 1):
    # 목표 가속도: -1 ~ 1 랜덤
    a_target = np.random.uniform(-1, 1)

    # 목표 물체 dynamics
    x_target_dot += a_target * dt  # 속도 업데이트
    x_target     += x_target_dot * dt  # 위치 업데이트
    target_pos[i+1] = x_target

    # 상대 상태 업데이트 (추적자 - 목표)
    e     = x_tracker     - x_target
    v_rel = x_tracker_dot - x_target_dot
    x = np.array([[e], [v_rel]])

    e_hist[i]     = e
    v_rel_hist[i] = v_rel

    # 제어 입력 (수치)
    # 상대계에서 dv_rel/dt = u - a_target 이므로, a_target을 feedforward로 상쇄
    u = a_target - float(K @ x)  # u는 스칼라
    u_hist[i] = u

    # 추적자 dynamics (dv_c/dt = u, dx_c/dt = v_c)
    x_tracker_dot += u * dt
    x_tracker     += x_tracker_dot * dt
    tracker_pos[i+1] = x_tracker

# 마지막 샘플의 상대 상태도 기록
e_hist[-1]     = x_tracker     - x_target
v_rel_hist[-1] = x_tracker_dot - x_target_dot

# =====================
# 도표 시각화
# =====================

# 1) 타겟 / 추적자 절대 위치
plt.figure(figsize=(8, 4))
plt.plot(t, target_pos, label="Target position")
plt.plot(t, tracker_pos, '--', label="Tracker position")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Target vs Tracker Position")
plt.legend()
plt.grid(True)

# 2) 상대 위치 / 상대 속도
plt.figure(figsize=(8, 4))
plt.plot(t, e_hist, label="Relative position e = x_tr - x_tg")
plt.plot(t, v_rel_hist, '--', label="Relative velocity v_rel = v_tr - v_tg")
plt.xlabel("Time [s]")
plt.ylabel("Relative state")
plt.title("Relative State (e, v_rel)")
plt.legend()
plt.grid(True)

# 3) 제어 입력
plt.figure(figsize=(8, 3))
plt.plot(t, u_hist)
plt.xlabel("Time [s]")
plt.ylabel("Control input u")
plt.title("Control Input Over Time")
plt.grid(True)

# =====================
# 1D 애니메이션 (실제로 움직이는 것 보기)
# =====================

fig_anim, ax_anim = plt.subplots(figsize=(10, 2))

xmin = min(np.min(tracker_pos), np.min(target_pos)) - 2.0
xmax = max(np.max(tracker_pos), np.max(target_pos)) + 2.0
ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(-0.5, 0.5)
ax_anim.set_xlabel("Position")
ax_anim.set_title("1D LQR Chaser-Target Animation")
ax_anim.grid(True)

# 기준선
ax_anim.axhline(y=0, color='black', linewidth=1)

# 점 두 개 (타겟, 추격자)
target_point,  = ax_anim.plot([], [], 'ro', markersize=10, label="Target")
tracker_point, = ax_anim.plot([], [], 'bo', markersize=10, label="Tracker")
ax_anim.legend()

def init():
    target_point.set_data([], [])
    tracker_point.set_data([], [])
    return target_point, tracker_point

def animate(frame):
    target_point.set_data([target_pos[frame]], [0])
    tracker_point.set_data([tracker_pos[frame]], [0])
    return target_point, tracker_point

anim = FuncAnimation(
    fig_anim,
    animate,
    init_func=init,
    frames=N,
    interval=10,   # ms 단위 (10이면 0.01초당 한 프레임)
    blit=True,
    repeat=True
)

plt.show()
