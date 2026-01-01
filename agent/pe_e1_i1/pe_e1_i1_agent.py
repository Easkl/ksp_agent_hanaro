from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

try:
    # Optional: only needed if you want true LQR (solve CARE)
    from scipy.linalg import solve_continuous_are
except Exception:
    solve_continuous_are = None

import kspdg.utils.utils as U

try:
    import krpc
except Exception:
    krpc = None

#관측로그
observation_history = []
observation_count = 0

_AGENT_KRPC_CONN = None
_AGENT_KRPC_SC = None


def _get_agent_spacecenter():
    """Best-effort kRPC connection from agent process.

    The environment runs in a different process and has its own kRPC connection.
    If we want to convert vectors into vessel body-frame inside the agent process,
    we need our own kRPC connection.
    """

    global _AGENT_KRPC_CONN, _AGENT_KRPC_SC

    if krpc is None:
        return None

    if _AGENT_KRPC_CONN is None:
        _AGENT_KRPC_CONN = krpc.connect(name="kspdg_agent")
        _AGENT_KRPC_SC = _AGENT_KRPC_CONN.space_center

    return _AGENT_KRPC_SC


def _accel_rhcbci_to_rhvbody(accel__rhcbci: np.ndarray) -> np.ndarray | None:
    """Convert a 3-vector from rhcbci to rhvbody (forward, right, down).

    Uses the same transformation steps as KSPDG env:
    rhcbci (RH) -> lhcbci -> lhvbody -> rhvbody
    """

    sc = _get_agent_spacecenter()
    if sc is None:
        return None

    vessel = sc.active_vessel
    if vessel is None:
        return None

    v__rhcbci = [float(accel__rhcbci[0]), float(accel__rhcbci[1]), float(accel__rhcbci[2])]
    v__lhcbci = U.convert_rhcbci_to_lhcbci(v__rhcbci=v__rhcbci)

    v__lhvbody = list(
        sc.transform_direction(
            direction=tuple(v__lhcbci),
            from_=vessel.orbit.body.non_rotating_reference_frame,
            to=vessel.reference_frame,
        )
    )
    v__rhvbody = U.convert_lhbody_to_rhbody(v__lhbody=v__lhvbody)
    return np.array(v__rhvbody, dtype=float)


def _compute_lqr_gain():
    if solve_continuous_are is None:
        return None

    a = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]).astype(float)
    r = np.diag([0.1, 0.1, 0.1]).astype(float)

    p = solve_continuous_are(a, b, q, r)
    k = np.linalg.solve(r, b.T @ p)
    return k


K_GAIN = _compute_lqr_gain()

# 최적제어 함수 정의
def optimal_control(observation):
    time = float(observation[0])
    vehicle_mass = float(observation[1])
    vehicle_propellant = float(observation[2])  # 지금은 안 쓰지만 그냥 남겨둬 봤음

    agent_x = float(observation[3])
    agent_y = float(observation[4])
    agent_z = float(observation[5])
    agent_vx = float(observation[6])
    agent_vy = float(observation[7])
    agent_vz = float(observation[8])

    target_x = float(observation[9])
    target_y = float(observation[10])
    target_z = float(observation[11])
    target_vx = float(observation[12])
    target_vy = float(observation[13])
    target_vz = float(observation[14])

    # 2) 상태벡터 x 정의 (agent - target 부호)
    error_x = agent_x - target_x
    error_y = agent_y - target_y
    error_z = agent_z - target_z
    error_vx = agent_vx - target_vx
    error_vy = agent_vy - target_vy
    error_vz = agent_vz - target_vz

    x = np.array([error_x, error_y, error_z, error_vx, error_vy, error_vz], dtype=float)

    # 3) xdot (최근 2개 observation 평균 변화율). dt가 너무 작으면 제외(None)
    xdot = None
    if len(observation_history) >= 2:
        prev_obs = observation_history[-2]
        dt = float(time - prev_obs[0])
        if abs(dt) > 1e-6:
            prev_agent_x = float(prev_obs[3])
            prev_agent_y = float(prev_obs[4])
            prev_agent_z = float(prev_obs[5])
            prev_agent_vx = float(prev_obs[6])
            prev_agent_vy = float(prev_obs[7])
            prev_agent_vz = float(prev_obs[8])
            prev_target_x = float(prev_obs[9])
            prev_target_y = float(prev_obs[10])
            prev_target_z = float(prev_obs[11])
            prev_target_vx = float(prev_obs[12])
            prev_target_vy = float(prev_obs[13])
            prev_target_vz = float(prev_obs[14])

            prev_x = np.array(
                [
                    prev_agent_x - prev_target_x,
                    prev_agent_y - prev_target_y,
                    prev_agent_z - prev_target_z,
                    prev_agent_vx - prev_target_vx,
                    prev_agent_vy - prev_target_vy,
                    prev_agent_vz - prev_target_vz,
                ],
                dtype=float,
            )
            xdot = (x - prev_x) / dt

    # 4) 제어입력 u 계산
    # K_GAIN이 None이면 여기서 크래시하므로, 최소한의 fallback 제공
    if K_GAIN is None:
        # 아주 약한 감쇠(속도만 줄이는) fallback
        kd = 0.05
        u_raw = np.array(
            [-kd * error_vx, -kd * error_vy, -kd * error_vz],
            dtype=float,
        )
    else:
        u_raw = -(K_GAIN @ x)

    # 포화/클립 없이 "순수" 필요 가속도만 사용
    u_accel = u_raw
    # (옵션) 디버그/해석용: rhcbci 기준 가속도를 agent body 기준으로 변환
    u_accel_agent = _accel_rhcbci_to_rhvbody(u_accel)

    # accel [m/s^2] -> thrust [N]
    thrust = vehicle_mass * u_accel

    burn_dur = 0.2
    action = {
        "burn_vec": [float(thrust[0]), float(thrust[1]), float(thrust[2]), float(burn_dur)],
        "ref_frame": 1,
        "vec_type": 1,
    }

    # 디버그 출력(검증용): 최적제어 식 검증에 필요한 최소 정보만
    print("x=", x)
    print("u_needed_accel=", u_accel)
    if u_accel_agent is not None:
        print("u_needed_accel_agent(rhvbody)=", u_accel_agent)
    
    return action, x, xdot

class PE1_E1_I1_Env_SAS(PE1_E1_I1_Env):
    def _reset_vessels(self):
        super()._reset_vessels()
        self.vesPursue.control.sas_mode = self.conn.space_center.SASMode.stability_assist #디폴트로 SAS가 target 모드로 설정돼 있길래 stability_assist로 바꿈

class PE_E1_I1_Agent(KSPDGBaseAgent):                                    
    def __init__(self):
        super().__init__()

    def get_action(self, observation): # Agent는 get_action만 무한반복하므로 추진 관련 코드는 get_action 안에 쓰기를 바란다
        global observation_count
        observation_history.append(observation)
        observation_count += 1
        action, x, xdot = optimal_control(observation)
        return action
    
if __name__ == "__main__":
    pe_e1_i1_agent = PE_E1_I1_Agent()    
    runner = AgentEnvRunner(
        agent=pe_e1_i1_agent, 
        env_cls=PE1_E1_I1_Env_SAS, 
        env_kwargs=None,
        runner_timeout=100,
        debug=False,
        enable_telemetry=False)
    print(runner.run())
