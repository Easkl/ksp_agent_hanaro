# Observation: [17.919999999999707, 6666.67431640625, 1195.41845703125, 716336.3616774438, -222098.8982851615, 30.337696723682875, 638.9452508032937, 2073.8918779282694, 5.146892685942836, 717141.6407256647, -219562.90047760372, -0.3836874253618774, 635.2619134334894, 2074.907781418355, 0.003621464993706908]
# 순서대로 agent_x, agent_y, agent_z, agent_vx, agent_vy, agent_vz, target_x, target_y, target_z, target_vx, target_vy, target_vz, fuel, time
# Navigation함수는 agent의 observation 배열을 기본 입력으로 받지만, 무조건 추적자와 도망자일 필요는 없다. 필요하다면 현재 위치와 목표 사이의 데이터를 매개변수로 전달해도 정상작동한다.
# 반환값은 agent가 목표지점에 도달하기 위해 필요한 가속도 벡터 [ax, ay, az]이다.

'''이 파일의 목적은 agent로부터 observation을 받아서 propulsion.py에 input값을 가속도 벡터 u로 전달해주는 것이다.'''

import numpy as np
from observation_log import observation_history
from scipy.linalg import solve_continuous_are

A_MATRIX = np.array([[0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])
B_MATRIX = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
Q_MATRIX = np.diag([.1, .1, .1, .5, .5, .5])
R_MATRIX = np.diag([.1, .1, .1])
P_MATRIX = solve_continuous_are(A_MATRIX, B_MATRIX, Q_MATRIX, R_MATRIX)
K_GAIN = np.linalg.inv(R_MATRIX) @ B_MATRIX.T @ P_MATRIX


def navigation(observation):
    time = observation[0]
    vehicle_mass = observation[1]   # 정의는 했지만 안쓰는 변수
    vehicle_propellant = observation[2]   # 정의는 했지만 안쓰는 변수
    agent_x = observation[3]
    agent_y = observation[4]
    agent_z = observation[5]
    agent_vx = observation[6]
    agent_vy = observation[7]
    agent_vz = observation[8]
    target_x = observation[9]
    target_y = observation[10]
    target_z = observation[11]
    target_vx = observation[12]
    target_vy = observation[13]
    target_vz = observation[14]

    error_x = target_x - agent_x
    error_y = target_y - agent_y
    error_z = target_z - agent_z
    rel_vx = target_vx - agent_vx
    rel_vy = target_vy - agent_vy
    rel_vz = target_vz - agent_vz

    #상태벡터 정의
    x = np.array([error_x, error_y, error_z, rel_vx, rel_vy, rel_vz])

    if len(observation_history) < 2:
        # observation_history에 충분한 데이터가 없으면 상대 가속도를 구할 수 없으므로 0으로 간주
        u = -K_GAIN @ x
        return u
    dt = observation_history[-1][0] - observation_history[-2][0]
    if abs(dt) < 1e-6:
        return -K_GAIN @ x  # 시간 차이가 사실상 0이면 가속도 보상 없이 피드백만 적용

    # 상대 가속도 계산
    rel_ax = ((observation_history[-1][12] - observation_history[-1][6]) - (observation_history[-2][12] - observation_history[-2][6])) / dt
    rel_ay = ((observation_history[-1][13] - observation_history[-1][7]) - (observation_history[-2][13] - observation_history[-2][7])) / dt
    rel_az = ((observation_history[-1][14] - observation_history[-1][8]) - (observation_history[-2][14] - observation_history[-2][8])) / dt
    rel_a = np.array([rel_ax, rel_ay, rel_az])
    u = -K_GAIN @ x + rel_a
    return u