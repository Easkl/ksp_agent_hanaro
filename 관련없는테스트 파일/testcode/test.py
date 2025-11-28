import casadi as ca
import numpy as np
import time

# 파라미터
N = 10  # horizon
dt = 0.1
T = N * dt

# 변수 정의 (상태 + 입력 + 보조)
opti = ca.Opti()
x1 = opti.variable(N+1)  # 위치 x
x2 = opti.variable(N+1)  # 속도 vx
x3 = opti.variable(N+1)  # 위치 y
x4 = opti.variable(N+1)  # 속도 vy
x5 = opti.variable(N+1)  # 위치 z
x6 = opti.variable(N+1)  # 속도 vz
x7 = opti.variable(N+1)  # 연료
x8 = opti.variable(N)    # 입력 ux
x9 = opti.variable(N)    # 입력 uy
x10 = opti.variable(N)   # 입력 uz
x11 = opti.variable(N+1) # 거리
x12 = opti.variable(N+1) # 속도 크기
x13 = opti.variable(N+1) # 연료 효율
x14 = opti.variable(N+1) # 제어 신호
x15 = opti.variable(N+1) # 에너지

# 행렬
A = np.eye(6)  # 단순화
B = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 0, 0],
              [0, 1, 0],
              [0, 0, 0],
              [0, 0, 1]])

# 초기 조건
opti.subject_to(x1[0] == 1)
opti.subject_to(x2[0] == 0)
opti.subject_to(x3[0] == 1)
opti.subject_to(x4[0] == 0)
opti.subject_to(x5[0] == 1)
opti.subject_to(x6[0] == 0)
opti.subject_to(x7[0] == 15)

# Dynamics (이산화)
for k in range(N):
    # ODE
    opti.subject_to(x1[k+1] == x1[k] + x2[k] * dt)
    opti.subject_to(x2[k+1] == x2[k] + (x8[k] + 0.01 * x1[k]**2) * dt)  # 비선형 줄임
    opti.subject_to(x3[k+1] == x3[k] + x4[k] * dt)
    opti.subject_to(x4[k+1] == x4[k] + (x9[k] + 0.01 * x3[k]**2) * dt)
    opti.subject_to(x5[k+1] == x5[k] + x6[k] * dt)
    opti.subject_to(x6[k+1] == x6[k] + (x10[k] + 0.01 * x5[k]**2) * dt)
    opti.subject_to(x7[k+1] == x7[k] - (ca.fabs(x8[k]) + ca.fabs(x9[k]) + ca.fabs(x10[k])) * dt)

    # 보조 변수
    opti.subject_to(x11[k] == ca.sqrt(x1[k]**2 + x3[k]**2 + x5[k]**2))
    opti.subject_to(x12[k] == ca.sqrt(x2[k]**2 + x4[k]**2 + x6[k]**2))
    opti.subject_to(x13[k] == ca.fmax(0.1, x7[k] / 15.0))
    opti.subject_to(x14[k] == ca.fmin(10, ca.fmax(-10, x12[k] * 0.5)))
    opti.subject_to(x15[k] == 0.5 * (x2[k]**2 + x4[k]**2 + x6[k]**2) + 0.1 * (x8[k]**2 + x9[k]**2 + x10[k]**2))

# 마지막 보조
opti.subject_to(x11[N] == ca.sqrt(x1[N]**2 + x3[N]**2 + x5[N]**2))
opti.subject_to(x12[N] == ca.sqrt(x2[N]**2 + x4[N]**2 + x6[N]**2))
opti.subject_to(x13[N] == ca.fmax(0.1, x7[N] / 15.0))
opti.subject_to(x14[N] == ca.fmin(10, ca.fmax(-10, x12[N] * 0.5)))
opti.subject_to(x15[N] == 0.5 * (x2[N]**2 + x4[N]**2 + x6[N]**2))

# 제약
opti.subject_to(opti.bounded(-10, x8, 10))
opti.subject_to(opti.bounded(-10, x9, 10))
opti.subject_to(opti.bounded(-10, x10, 10))
opti.subject_to(x7 >= 0)

# 목적 함수
opti.minimize(ca.sum1(x11) + ca.sum1(x15) - 0.1 * ca.sum1(x7))

# 시간 측정 시작
start_time = time.time()

# 풀기
opti.solver('ipopt')
sol = opti.solve()

# 시간 측정 끝
end_time = time.time()

# 결과 출력
print("최종 x1:", sol.value(x1[-1]))
print("최종 x3:", sol.value(x3[-1]))
print("최종 x5:", sol.value(x5[-1]))
print("최종 연료:", sol.value(x7[-1]))
print("최종 거리:", sol.value(x11[-1]))
print("최종 에너지:", sol.value(x15[-1]))
print(f"계산 시간: {end_time - start_time:.2f} 초")