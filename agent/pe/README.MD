# PE Agent

## 배경: PE는 16가지 시나리오

PE는 총 16가지의 경우가 있습니다. `e1~e4`(evader 정책 4종)와 `i1~i4`(초기 조건 4종)를 조합해서 16가지 상황을 만들 수 있습니다.

저는 16가지 상황 모두에서 그럴듯한 결과가 나오도록 **중복을 피해서 하나의 agent 코드**로 정리했습니다. 다만 시나리오별로 환경 클래스가 다르기 때문에, 실행하려는 상황에 맞게 **코드 앞부분의 import(환경 import)** 를 바꿔줘야 정상 동작한다는 점을 알아두면 좋습니다.

예를 들어, 현재 문서는 `pe_e1_i1_agent.py` 기준이며 코드 상단의 아래 줄이 시나리오 선택 지점입니다.

```python
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
```

다른 케이스로 돌리려면 (예: `E1_I2`) 위 import를 `PE1_E1_I2_Env`로 바꾸고, 코드 내 `ENV_CLS = ...`도 같은 환경 클래스로 맞춰주면 됩니다.

---

## 목차

- 파일 위치
- 실행 환경 / 의존성
- 관측(observation) 벡터 구조
- 액션(action) 딕셔너리 구조
- 좌표계(rhcbci / rhvbody)와 이 코드의 선택
- 코드 구성요소 설명(위에서 아래 순서)
- 핵심 로직: `optimal_control(observation)`
- 현실 보정 1: `ENABLE_BRAKE_OVERRIDE` (포화 + 히스테리시스)
- 현실 보정 2: Cross-track(측면) 억제
- 최종 액션 패킹
- 에이전트 클래스: `PE_E1_I1_Agent`
- 실행부: `__main__`
- 기대 결과(무엇이 “잘 된 것”인가?)
- 튜닝 포인트(성능이 애매할 때 조절할 값)

이 문서는 `pe_e1_i1_agent.py`를 처음 보는 사람도 **왜 이런 코드를 썼는지(의도)**, **어떻게 동작하는지(구조/흐름)**, **무슨 결과를 기대하는지(행동/성능)**를 이해할 수 있도록, 코드의 구성 요소를 위에서 아래 순서대로 설명합니다.

> 요약
>
> - 환경: `kspdg.pe1.e1_envs.PE1_E1_I1_Env` (evader = e1: 기동 없음)
> - 목표: pursuer가 evader(타겟)에 **가까이 접근**하고 **상대속도도 줄이는** rendezvous/pursuit
> - 방법: (가능하면) 연속시간 LQR로 “필요한 가속도”를 계산 → 질량을 곱해 “필요한 추력[N]”으로 변환
> - 현실 보정: RCS 최대추력 제약을 반영하는 **브레이크(포화) 로직 + 히스테리시스**로 과도한 오버슈트/핑퐁 억제
> - 추가 보정: 매우 작은 측면 오차에서 RCS 채터(chatter) 억제

---

## 실행 환경 / 의존성

이 에이전트는 `spacegym-kspdg-main`의 환경을 이용합니다.

- 필수: `kspdg` (이 레포의 `spacegym-kspdg-main/src`)
- 선택(있으면 더 “제대로” 동작):
  - `scipy` : LQR 이득 계산에 사용 (`solve_continuous_are`)
  - `krpc` : 좌표계 변환 디버그/보조 변환에 사용 (inertial → vessel body)

> 참고
>
> - `scipy`가 없으면 LQR 대신 **상대속도 감쇠(kd)**만으로 추력을 계산합니다(성능은 떨어질 수 있음).
> - `krpc`가 없으면 agent 측 변환 함수가 `None`을 반환하지만, 액션을 `ref_frame=1`(rhcbci)로 보낼 수 있어 최소한의 호환을 유지합니다.

---

## 관측(observation) 벡터 구조

PE1 그룹의 관측은 `kspdg.pe1.pe1_base.PursuitEvadeGroup1Env`에 정의되어 있고 길이는 15입니다.

|  index | 의미                                         | 단위 |
| -----: | -------------------------------------------- | ---- |
|      0 | mission elapsed time                         | s    |
|      1 | pursuer(내 기체) mass                        | kg   |
|      2 | pursuer propellant mass (mono prop)          | kg   |
|   3..5 | pursuer position wrt celestial body (rhcbci) | m    |
|   6..8 | pursuer velocity wrt celestial body (rhcbci) | m/s  |
|  9..11 | evader position wrt celestial body (rhcbci)  | m    |
| 12..14 | evader velocity wrt celestial body (rhcbci)  | m/s  |

즉, 코드에서 `agent_*`는 pursuer, `target_*`는 evader를 의미합니다.

---

## 액션(action) 딕셔너리 구조

Group1 환경(`kspdg.base_envs.Group1BaseEnv.vessel_step`)은 다음 형식의 딕셔너리를 받습니다.

```python
action = {
	"burn_vec": [x, y, z, burn_dur],
	"ref_frame": 0|1|2,
	"vec_type": 0|1,
}
```

- `burn_vec[0:3]`: 3축 벡터
- `burn_vec[3]`: 연소(추력 적용) 시간 [s]
- `ref_frame`:
  - `0` = `rhvbody` (vessel body: forward, right, down)
  - `1` = `rhcbci` (천체 중심 관성 좌표)
  - `2` = `rhntw` (궤도 NTW 좌표)
- `vec_type`:
  - `0` = throttle ([-1,1])
  - `1` = thrust [N]

이 에이전트는 **`vec_type=1`(추력[N])**을 사용합니다. 이유는 “가속도 요구량 → 질량×가속도 = 추력” 변환을 에이전트가 직접 수행하고, 환경은 그 추력을 RCS 최대추력으로 나눠 throttle로 변환하게 하기 위함입니다.

---

## 좌표계(rhcbci / rhvbody)와 이 코드의 선택

- 관측은 항상 `rhcbci` (관성 좌표)로 들어옵니다.
- 실제 RCS 제어는 “기체 바디축” 기준으로 적용되는 게 자연스럽습니다.

그래서 이 코드는 다음 두 경우를 모두 지원합니다.

1. **가능하면 body 축에서 브레이크/포화 판단** (`krpc` 변환 성공 시)
   - `ref_frame=0`로 액션을 내보냄
2. 변환이 불가능하면 **그냥 관성좌표(rhcbci)로 액션을 내보냄**
   - `ref_frame=1`

---

## 코드 구성요소 설명(위에서 아래 순서)

### 1) import / optional import

```python
from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner
import numpy as np
```

- 에이전트 베이스 클래스, 환경 클래스, 실행 러너를 가져옵니다.

```python
try:
		from scipy.linalg import solve_continuous_are
except Exception:
		solve_continuous_are = None
```

- `scipy`가 있으면 LQR 이득을 계산하고, 없으면 `None`으로 두어 fallback 제어기로 전환합니다.

```python
try:
		import krpc
except Exception:
		krpc = None
```

- `krpc`가 있으면 agent 측에서도 좌표 변환을 시도합니다(주로 body축 기반 제어/디버그용).

---

### 2) 전역 상태: 히스토리/브레이크 상태

```python
observation_history = []
observation_count = 0
```

- 이전 관측을 저장해서 `xdot`(상태 미분)를 근사 계산할 때 사용합니다.
- 현재 `observation_count`는 카운트만 하고 실제 로직에는 사용하지 않습니다.

```python
_BRAKE_AXIS_STATE = np.array([False, False, False], dtype=bool)
```

- 각 축별로 “브레이크 모드(최대 역추력으로 감속)”가 켜졌는지 저장합니다.
- 히스테리시스를 위해 필요합니다(켜짐/꺼짐 기준이 다름).

---

### 3) RCS 및 브레이크/채터 억제 파라미터

이것이 존재하는 이유는 다음과 같습니다.

이 agent는 MPC가 아닌 LQR알고리즘을 기반으로 설계했습니다. 그렇기 때문에 입력 요구값이 RCS의 스펙을 넘어서는 수치가 나오면 evader를 지나쳐버리는 현상이 발생합니다. 따라서 이를 방지하기 위해 지금 접근하는 속도의 retrogade방향으로 최대한 가속을 했을때 멈추는 거리 어쩌고 저쩌고 여기는 좀 자세히 설명을 하긴 해야돼

```python
RCS_THRUST_PER_THRUSTER_N = 1000.0
RCS_THRUSTER_COUNT_PER_AXIS = np.array([8.0, 4.0, 4.0], dtype=float)
```

- 축별 최대추력을 추정하기 위한 상수입니다.
- `[forward, right, down]` 축에서 각각 8/4/4개의 RCS가 유효하다고 가정합니다.(실제로 시뮬레이션을 돌려보면 x축으로는 8개의 RCS가 작동하고, y, z축으로는 4개의 RCS가 작동합니다)

```python
CROSS_TRACK_POS_DEADBAND_M = 5.0
CROSS_TRACK_VEL_DEADBAND_MPS = 0.10
CROSS_TRACK_THRUST_SCALE = 0.25
```

- 측면(right/down) 오차가 아주 작을 때 thruster chatter를 줄이기 위한 deadband/스케일입니다.

```python
BRAKE_ON_FACTOR = 0.4
BRAKE_OFF_FACTOR = 0.15
BRAKE_V_RELEASE = 0.02
```

- 브레이크 모드 히스테리시스:
  - `ON`: “이 속도로는 지금 거리에서 못 멈춘다” 판단을 더 일찍 하도록(보수적으로) 설정
  - `OFF`: 속도가 충분히 작아지거나 안전 영역이면 브레이크 해제

---

### 4) agent 측 kRPC 연결 및 좌표 변환 도우미

```python
def _get_agent_spacecenter():
		...
```

- `krpc`가 있을 때만, agent 이름 `kspdg_agent`로 별도 연결을 잡고 `space_center`를 리턴합니다.

```python
def _accel_rhcbci_to_rhvbody(accel__rhcbci):
		...
```

- 관성좌표(rhcbci)의 3벡터를 vessel body(rhvbody)로 바꿉니다.
- 내부적으로:
  1. `rhcbci → lhcbci` 변환 (kspdg util)
  2. `transform_direction(from=non_rotating_reference_frame, to=vessel.reference_frame)`
  3. `lhvbody → rhvbody` 변환

> 왜 agent 측 변환이 필요한가?
>
> - “브레이크(포화) 판단”을 실제 RCS 적용축과 맞추면 제어가 더 안정적이고 직관적입니다.

---

### 5) LQR 이득 계산

```python
def _compute_lqr_gain():
		...
```

- 상태를 `x = [pos_error(3), vel_error(3)]`로 두고, 입력을 `u = accel(3)`로 둔 **3D double integrator** 모델을 사용합니다.

연속시간 상태공간은 다음과 같습니다.

$$
\dot{x} = Ax + Bu
$$

여기서

- $A$는 상단 우측에 I(속도가 위치 미분)만 있고 나머지 0
- $B$는 가속도가 속도 미분에 직접 들어가도록 구성

가중치 행렬은:

- $Q = diag([0.1,0.1,0.1,\;0.5,0.5,0.5])$ (속도 오차를 더 강하게 벌점)
- $R = diag([0.1,0.1,0.1])$ (입력도 어느 정도 벌점)

그리고

$$
u = -Kx
$$

가 되도록 CARE를 풀어 $K$를 계산합니다.

```python
K_GAIN = _compute_lqr_gain()
```

- 실행 시 한 번 계산된 이득을 전역으로 재사용합니다.

---

### 6) 환경 선택

```python
ENV_CLS = PE1_E1_I1_Env
```

- 현재 파일은 PE1 E1 I1 시나리오(초기조건 i1, evader 기동 없음)를 대상으로 합니다.

---

## 핵심 로직: `optimal_control(observation)`

이 함수가 “관측 → 제어 입력 계산 → 액션 딕셔너리 생성” 전체를 담당합니다.

### 1) 관측 파싱

```python
time = observation[0]
vehicle_mass = observation[1]
...
agent_x..agent_vz = observation[3:9]
target_x..target_vz = observation[9:15]
```

### 2) 상대 상태(state) 구성

```python
error_x = agent_x - target_x
...
x = [pos_error, vel_error]
```

- pursuer가 evader에 대해 얼마나 떨어져 있고 얼마나 빠르게 상대운동하는지 나타내는 상태입니다.

### 3) 에피소드 리셋 감지(브레이크 상태 초기화)

```python
if time < prev_time:
		_BRAKE_AXIS_STATE[:] = False
```

- 환경이 리셋되면 time이 뒤로 갈 수 있으므로, 축별 브레이크 상태를 초기화합니다.

### 4) 상태 미분 `xdot` 근사(디버그/분석용)

```python
xdot = (x - prev_x) / dt
```

- 현재 로직에서 `xdot`는 제어에는 직접 쓰지 않고, 관찰/디버깅용으로 함께 반환합니다.

### 5) “필요 가속도” 계산 (LQR 또는 fallback)

```python
if K_GAIN is None:
		u_raw = -kd * vel_error
else:
		u_raw = -(K_GAIN @ x)
```

- LQR이 있으면 위치/속도 오차를 동시에 줄이는 방향으로 가속도를 생성합니다.
- LQR이 없으면 속도만 감쇠시키는 단순 제어기로 최소 동작을 보장합니다.

여기서 `u_raw`는 관성좌표(rhcbci) 기준의 “가속도”로 취급됩니다.

### 6) 좌표 변환 시도 (가능하면 body축)

```python
u_accel_agent = _accel_rhcbci_to_rhvbody(u_accel)
```

- 변환이 성공하면 `rhvbody`로도 같은 벡터를 갖게 됩니다.

### 7) “필요 추력[N]”으로 변환

```python
thrust = vehicle_mass * accel
```

- 환경은 추력(N)을 throttle로 변환할 수 있으므로, 에이전트가 질량을 곱해 추력 형태로 명령합니다.

---

## 현실 보정 1: `ENABLE_BRAKE_OVERRIDE` (포화 + 히스테리시스)

LQR이 계산한 추력은 “이상적”인 연속 입력이기 때문에, 실제 RCS의 최대추력 한계를 넘길 수 있고, 그 결과 오버슈트/진동이 생길 수 있습니다.

그래서 이 코드는 각 축별로 “지금 거리에서 이 속도로는 못 멈춘다”를 판단하면 **최대 역추력으로 브레이크**를 걸어줍니다.

### 1) 최대추력 및 최대가속도 계산

```python
max_thrust_n = count_per_axis * thrust_per_thruster
a_max = max_thrust_n / mass
```

### 2) body축 기반 브레이크를 우선

- 변환이 성공하면 `pos_err_body`, `vel_err_body`를 사용합니다.
- 실패하면 관성좌표 오차를 사용합니다.

### 3) 축별 stop-distance 기반 브레이크 조건

```python
stop_limit = 2 * a_max * abs(pos_err)
if vel^2 > BRAKE_ON_FACTOR * stop_limit:
		brake = ON
```

- $v^2 \le 2 a \Delta x$ 형태의 에너지/등가속 감속 조건을 이용한 근사 판단입니다.
- 히스테리시스 때문에 ON/OFF 기준이 다릅니다.

### 4) 브레이크가 켜지면 해당 축을 “최대 역추력”으로 고정

```python
thrust[i] = -sign(vel_err[i]) * max_thrust_n[i]
```

- 속도의 부호와 반대로 최대추력을 걸어 감속을 최우선합니다.

---

## 현실 보정 2: Cross-track(측면) 억제

body축을 쓸 수 있을 때만 적용됩니다.

- `i=1,2` (right, down) 축에 대해:
  - 오차가 아주 작으면 아예 0으로 만들어 채터 제거
  - 그렇지 않으면 측면 추력을 `0.25`로 줄여, forward 축 제어를 상대적으로 더 우선

---

## 최종 액션 패킹

```python
burn_dur = 0.2
action = {
	"burn_vec": [Fx, Fy, Fz, burn_dur],
	"ref_frame": 0 or 1,
	"vec_type": 1,
}
```

- body 변환이 가능하면 `ref_frame=0`(rhvbody)로 내보냅니다.
- 아니면 `ref_frame=1`(rhcbci)로 내보냅니다.

그리고 디버깅을 위해 다음을 출력합니다.

- `x` (상대 상태)
- `u_needed_accel` (관성좌표 가속도 요구)
- `u_needed_accel_agent` (가능하면 body축 가속도 요구)

---

## 에이전트 클래스: `PE_E1_I1_Agent`

```python
class PE_E1_I1_Agent(KSPDGBaseAgent):
		def get_action(self, observation):
				observation_history.append(observation)
				action, x, xdot = optimal_control(observation)
				return action
```

- 프레임워크가 호출하는 `get_action()`에서 관측을 히스토리에 저장하고, `optimal_control()`의 결과 중 **action만** 반환합니다.

---

## 실행부: `__main__`

```python
runner = AgentEnvRunner(
	agent=pe_e1_i1_agent,
	env_cls=PE1_Env_SAS,
	...
)
print(runner.run())
```

- `AgentEnvRunner`가 환경을 만들고, step 루프를 돌면서 관측을 전달하고 액션을 적용합니다.

`PE1_Env_SAS`는 환경 리셋 시 SAS 모드를 제어할 수 있도록 확장한 클래스입니다(현재 stability_assist 설정은 주석 처리되어 있습니다).

### 실행 방법(간단)

1. (필요 시) `kspdg`를 editable로 설치

```bash
pip install -e spacegym-kspdg-main
```

2. 에이전트 실행

```bash
python agent/pe/pe_e1_i1_agent.py
```

> 주의
>
> - 이 환경은 kRPC/KSP 실행 상태에 의존합니다. (환경 세팅은 `spacegym-kspdg-main/README.md`의 설치/실행 가이드를 따르는 것을 권장합니다. 그리고 해봤는데 현재 이 문서를 작성하고 있는 날짜인 2026-01-08 기준으로 최신 파이썬 버전은 3.14.2입니다. 그러나 이 버전의 파이썬으로는 MIT의 KSPDG의 설치가 제대로 동작하지 않았습니다. 따라서 버전을 다운그레이드하여 실행하시는 것이 좋을 것 같습니다.)

---

## 기대 결과(무엇이 “잘 된 것”인가?)

PE1의 평가지표는 대체로:

- 가장 가까운 접근 거리(closest approach distance)
- 그 때의 상대속도
- 연료 사용량
- 걸린 시간

으로 구성됩니다.

이 에이전트의 의도는:

1. LQR로 상대 위치/속도를 동시에 줄이되,
2. 실제 RCS 제약 때문에 생기는 오버슈트/진동을 브레이크 히스테리시스로 억제하고,
3. 아주 작은 측면 오차에서 불필요한 RCS 채터를 줄여 연료/안정성을 개선

을 통해 위 지표를 전반적으로 개선하는 것입니다.

환경 관점에서 가장 직관적인 “성공”은 pursuer가 evader에 **충분히 가까이 접근(capture)** 하는 것입니다. PE1 기본값은 `capture_dist = 5.0 m`이며(시나리오 설정에 따라 달라질 수 있음), 여기에 더해 접근 시점의 상대속도/연료/시간이 작을수록 점수가 좋아집니다.

---

## 튜닝 포인트(성능이 애매할 때 조절할 값)

- LQR 가중치:
  - `_compute_lqr_gain()`의 `Q`, `R`
  - 속도 오차를 더 줄이고 싶으면 `Q`의 속도 항을 키우거나 `R`을 줄입니다.
- 브레이크 히스테리시스:
  - `BRAKE_ON_FACTOR` 낮추면 더 일찍 브레이크(더 보수적)
  - `BRAKE_OFF_FACTOR`, `BRAKE_V_RELEASE` 낮추면 더 오래 브레이크 유지
- 측면 억제:
  - `CROSS_TRACK_*` 값으로 deadband/스케일 조절
- 액션 시간:
  - `burn_dur = 0.2` (짧을수록 더 자주 피드백, 너무 짧으면 계산/통신 부하)
