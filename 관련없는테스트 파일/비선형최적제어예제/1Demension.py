import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# ====== 초기 조건 ======
tracker_x = 0.0
tracker_v = 0.0

target_x = 20.0
target_v = 0.0

# 시뮬레이션 설정
T = 20.0
dt = 0.01
N = int(T / dt)
t = np.linspace(0, T, N)

# ====== 상태벡터 정의 ======
# error = tracker - target
# rel_v = v_tracker - v_target
error = tracker_x - target_x
rel_v = tracker_v - target_v
x = np.array([error, rel_v])  # [e, e_dot]

# ====== LQR 설계 (연속시간) ======
# 상태: [e, e_dot]
A = np.array([[0., 1.],
              [0., 0.]])
B = np.array([[0.],
              [1.]])  # 입력 u = tracker 가속도

Q = np.array([[10., 0.],
              [0.,  1.]])   # 위치 에러를 더 강하게 벌점
R = np.array([[1.]])         # 입력 크기 벌점

P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P   # 1x2

print("LQR gain K:", K)

# ====== 기록용 배열 ======
tracker_x_hist = np.zeros(N)
target_x_hist = np.zeros(N)
error_hist = np.zeros(N)
u_hist = np.zeros(N)

tracker_x_hist[0] = tracker_x
target_x_hist[0] = target_x
error_hist[0] = error

# ====== 시뮬레이션 루프 ======
for i in range(N-1):
    # 1) 타겟 가속도 (외란) - 너무 크지 않게 적당히
    a_target = np.random.uniform(-1.0, 1.0)

    # 2) 타겟 dynamics 업데이트
    target_v += a_target * dt
    target_x += target_v * dt

    # 3) 상대 상태 업데이트 (tracker - target 기준)
    error = tracker_x - target_x
    rel_v = tracker_v - target_v
    x = np.array([error, rel_v])

    # 4) 제어 입력: u = -Kx + a_target
    #    (여기서 a_target은 "외란 보상"용 feedforward)
    u = float(-K @ x + a_target)
    
    # 5) 추적자 dynamics 업데이트
    tracker_v += u * dt
    tracker_x += tracker_v * dt

    # 6) 기록
    tracker_x_hist[i+1] = tracker_x
    target_x_hist[i+1] = target_x
    error_hist[i+1] = error
    u_hist[i+1] = u

# ====== 결과 플롯 ======

plt.figure(figsize=(10, 6))
plt.plot(t, target_x_hist, label="Target position", linewidth=2)
plt.plot(t, tracker_x_hist, label="Tracker position", linewidth=2, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Position (1D)")
plt.title("1D LQR Tracker vs Target")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, error_hist, label="error = tracker - target")
plt.xlabel("Time [s]")
plt.ylabel("Error")
plt.title("Position Error Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, u_hist, label="Control input u")
plt.xlabel("Time [s]")
plt.ylabel("u (tracker acceleration)")
plt.title("Control Input Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
