from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np

try:
    from scipy.linalg import solve_continuous_are
except Exception:
    solve_continuous_are = None

import kspdg.utils.utils as U

try:
    import krpc
except Exception:
    krpc = None

observation_history = []
observation_count = 0

# RCS max thrust per axis [N] = count * per-thruster thrust
RCS_THRUST_PER_THRUSTER_N = 1000.0
RCS_THRUSTER_COUNT_PER_AXIS = np.array([8.0, 4.0, 4.0], dtype=float)

_BRAKE_AXIS_STATE = np.array([False, False, False], dtype=bool)

_AGENT_KRPC_CONN = None
_AGENT_KRPC_SC = None


def _get_agent_spacecenter():
    """Best-effort agent-side kRPC connection (for frame conversion debug)."""

    global _AGENT_KRPC_CONN, _AGENT_KRPC_SC

    if krpc is None:
        return None

    if _AGENT_KRPC_CONN is None:
        _AGENT_KRPC_CONN = krpc.connect(name="kspdg_agent")
        _AGENT_KRPC_SC = _AGENT_KRPC_CONN.space_center

    return _AGENT_KRPC_SC


def _accel_rhcbci_to_rhvbody(accel__rhcbci: np.ndarray) -> np.ndarray | None:
    """Convert a 3-vector from rhcbci to rhvbody (forward, right, down)."""

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

def optimal_control(observation):
    global _BRAKE_AXIS_STATE

    time = float(observation[0])
    vehicle_mass = float(observation[1])
    vehicle_propellant = float(observation[2])

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

    error_x = agent_x - target_x
    error_y = agent_y - target_y
    error_z = agent_z - target_z
    error_vx = agent_vx - target_vx
    error_vy = agent_vy - target_vy
    error_vz = agent_vz - target_vz

    x = np.array([error_x, error_y, error_z, error_vx, error_vy, error_vz], dtype=float)

    if len(observation_history) >= 2:
        prev_time = float(observation_history[-2][0])
        if time + 1e-9 < prev_time:
            _BRAKE_AXIS_STATE[:] = False

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

    if K_GAIN is None:
        kd = 0.05
        u_raw = np.array(
            [-kd * error_vx, -kd * error_vy, -kd * error_vz],
            dtype=float,
        )
    else:
        u_raw = -(K_GAIN @ x)

    u_accel = u_raw
    u_accel_agent = _accel_rhcbci_to_rhvbody(u_accel)

    thrust = vehicle_mass * u_accel

    ENABLE_BRAKE_OVERRIDE = True
    if ENABLE_BRAKE_OVERRIDE:
        max_thrust_n = RCS_THRUSTER_COUNT_PER_AXIS * RCS_THRUST_PER_THRUSTER_N
        a_max = max_thrust_n / max(vehicle_mass, 1e-9)

        pos_err = x[0:3]
        vel_err = x[3:6]
        brake_on_factor = 1.0
        brake_off_factor = 0.6
        v_release = 0.15

        for i in range(3):
            stop_limit = 2.0 * a_max[i] * abs(pos_err[i]) + 1e-12
            v2 = vel_err[i] ** 2

            if not _BRAKE_AXIS_STATE[i]:
                if v2 > brake_on_factor * stop_limit:
                    _BRAKE_AXIS_STATE[i] = True
            else:
                if (abs(vel_err[i]) < v_release) or (v2 < brake_off_factor * stop_limit):
                    _BRAKE_AXIS_STATE[i] = False

            if _BRAKE_AXIS_STATE[i]:
                s = 1.0 if vel_err[i] >= 0.0 else -1.0
                thrust[i] = -s * max_thrust_n[i]

    burn_dur = 0.2
    action = {
        "burn_vec": [float(thrust[0]), float(thrust[1]), float(thrust[2]), float(burn_dur)],
        "ref_frame": 1,
        "vec_type": 1,  # thrust [N]
    }

    print("x=", x)
    print("u_needed_accel=", u_accel)
    if u_accel_agent is not None:
        print("u_needed_accel_agent(rhvbody)=", u_accel_agent)
    
    return action, x, xdot

class PE1_E1_I1_Env_SAS(PE1_E1_I1_Env):
    def _reset_vessels(self):
        super()._reset_vessels()
        # self.vesPursue.control.sas_mode = self.conn.space_center.SASMode.stability_assist

class PE_E1_I1_Agent(KSPDGBaseAgent):                                    
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
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
