from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# GEKKO 모델 생성
m = GEKKO()

# 시간 설정 (0~10초, 101 포인트)
m.time = np.linspace(0, 10, 101)

# 변수 10개 (상태 변수들)
x1 = m.Var(value=1)  # 위치 1
x2 = m.Var(value=0)  # 속도 1
x3 = m.Var(value=1)  # 위치 2
x4 = m.Var(value=0)  # 속도 2
x5 = m.Var(value=1)  # 위치 3
x6 = m.Var(value=0)  # 속도 3
x7 = m.Var(value=10) # 연료
x8 = m.Var(value=0, lb=-5, ub=5) # 입력 1
x9 = m.Var(value=0, lb=-5, ub=5) # 입력 2
x10 = m.Var(value=0, lb=-5, ub=5) # 입력 3

# 비선형 ODE (dynamics)
m.Equation(x1.dt() == x2)  # dx1/dt = x2
m.Equation(x2.dt() == x8 / 1.0 + 0.1 * x1**2)  # dx2/dt = u1 + 비선형 항
m.Equation(x3.dt() == x4)
m.Equation(x4.dt() == x9 / 1.0 + 0.1 * x3**2)
m.Equation(x5.dt() == x6)
m.Equation(x6.dt() == x10 / 1.0 + 0.1 * x5**2)

# 연료 소모 (ODE)
m.Equation(x7.dt() == -m.abs(x8) - m.abs(x9) - m.abs(x10))  # df/dt = -|u1| - |u2| - |u3|

# 대수 방정식 (제약/관계)
m.Equation(x8 == m.max2(-5, m.min2(5, x2 * 2)))  # u1 = saturate(2*x2)
m.Equation(x9 == m.max2(-5, m.min2(5, x4 * 2)))
m.Equation(x10 == m.max2(-5, m.min2(5, x6 * 2)))

# 초기 조건
m.fix_initial(x1, 1)
m.fix_initial(x2, 0)
m.fix_initial(x3, 1)
m.fix_initial(x4, 0)
m.fix_initial(x5, 1)
m.fix_initial(x6, 0)
m.fix_initial(x7, 10)

# 목적 함수 (연료 최소화 + 최종 위치 목표)
m.Minimize(m.integral(x7) + (x1 - 5)**2 + (x3 - 5)**2 + (x5 - 5)**2)

# 시뮬레이션 모드
m.options.IMODE = 6  # 최적화
m.options.NODES = 3   # 속도 향상
m.solve()

# 결과 출력
print("Final x1:", x1.value[-1])
print("Final x3:", x3.value[-1])
print("Final x5:", x5.value[-1])
print("Final fuel:", x7.value[-1])

# 플롯
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(m.time, x1.value, label='x1')
plt.plot(m.time, x3.value, label='x3')
plt.plot(m.time, x5.value, label='x5')
plt.legend()
plt.title('Positions')

plt.subplot(2, 2, 2)
plt.plot(m.time, x2.value, label='v1')
plt.plot(m.time, x4.value, label='v2')
plt.plot(m.time, x6.value, label='v3')
plt.legend()
plt.title('Velocities')

plt.subplot(2, 2, 3)
plt.plot(m.time, x7.value, label='Fuel')
plt.legend()
plt.title('Fuel')

plt.subplot(2, 2, 4)
plt.plot(m.time, x8.value, label='u1')
plt.plot(m.time, x9.value, label='u2')
plt.plot(m.time, x10.value, label='u3')
plt.legend()
plt.title('Inputs')

plt.tight_layout()
plt.show()