# Observation: [17.919999999999707, 6666.67431640625, 1195.41845703125, 716336.3616774438, -222098.8982851615, 30.337696723682875, 638.9452508032937, 2073.8918779282694, 5.146892685942836, 717141.6407256647, -219562.90047760372, -0.3836874253618774, 635.2619134334894, 2074.907781418355, 0.003621464993706908]
# 순서대로 agent_x, agent_y, agent_z, agent_vx, agent_vy, agent_vz, target_x, target_y, target_z, target_vx, target_vy, target_vz, fuel, time

from numpy import np

# Navigation함수는 agent의 observation 배열을 기본 입력으로 받지만, 무조건 추적자와 도망자일 필요는 없다. 필요하다면 현재 위치와 목표 사이의 데이터를 매개변수로 전달해도 정상작동한다.
# 반환값은 agent가 목표지점에 도달하기 위해 필요한 가속도 벡터 [ax, ay, az]이다.
def navigation(observation):
    T = 600.0
    t = 0.1
    N = int(T / t)

    time = observation[0]
    vehicle_mass = observation[1]
    vehicle_propellant = observation[2]
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

    rel_ax = 1
    rel_ay = 2
    rel_az = 3
    rel_a = np.array([rel_ax, rel_ay, rel_az])

    #상태벡터 정의
    stateVector = np.array([error_x, error_y, error_z, rel_vx, rel_vy, rel_vz])
    #최적제어 행렬 정의
    K = np.array([[0.1, 0, 0, 0.2, 0, 0],
                  [0, 0.1, 0, 0, 0.2, 0],
                  [0, 0, 0.1, 0, 0, 0.2]])
    u = -K @ stateVector + rel_a

    A = np.array([[1, 0, 0, t, 0, 0],
                [0, 1, 0, 0, t, 0],
                [0, 0, 1, 0, 0, t],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
    B = np.array([[0.5 * t**2, 0, 0],
                [0, 0.5 * t**2, 0],
                [0, 0, 0.5 * t**2],
                [t, 0, 0],
                [0, t, 0],
                [0, 0, t]])

    Q = np.diag([100, 100, 100, 10, 10, 10])
    R = np.diag([1, 1, 1])

# 이전 프레임 속도 저장용 (초기값 0)
prev_agent_vx = 0
prev_agent_vy = 0
prev_agent_vz = 0
prev_target_vx = 0
prev_target_vy = 0
prev_target_vz = 0

def compute_acceleration(current_v, prev_v, dt):
    return (current_v - prev_v) / dt

# 예시: get_action 또는 update 함수 내부에서 사용
# dt는 시뮬레이션 시간 간격

def navigation_step(observation, dt):
    global prev_agent_vx, prev_agent_vy, prev_agent_vz, prev_target_vx, prev_target_vy, prev_target_vz
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

    # 가속도 계산
    agent_ax = compute_acceleration(agent_vx, prev_agent_vx, dt)
    agent_ay = compute_acceleration(agent_vy, prev_agent_vy, dt)
    agent_az = compute_acceleration(agent_vz, prev_agent_vz, dt)
    target_ax = compute_acceleration(target_vx, prev_target_vx, dt)
    target_ay = compute_acceleration(target_vy, prev_target_vy, dt)
    target_az = compute_acceleration(target_vz, prev_target_vz, dt)

    # 이전 속도 업데이트
    prev_agent_vx = agent_vx
    prev_agent_vy = agent_vy
    prev_agent_vz = agent_vz
    prev_target_vx = target_vx
    prev_target_vy = target_vy
    prev_target_vz = target_vz

    # 상대 가속도
    rel_ax = agent_ax - target_ax
    rel_ay = agent_ay - target_ay
    rel_az = agent_az - target_az

    return rel_ax, rel_ay, rel_az
