'''이 파일에서 해야 하는것 navigation.py의 navigation함수에서 반환한 가속도 벡터를
받아서 rcs의 분사 벡터로 전환해서 크기 3의 array를 반환해야 한다'''
import numpy as np


def propulsion(accelration_vector, observation, vessel=None):
    accelration_vector = np.asarray(accelration_vector, dtype=float)
    if vessel is not None:
        ref_frame = vessel.orbit.body.reference_frame
        transformed = vessel.transform_direction(tuple(accelration_vector), ref_frame, vessel.reference_frame)
        accelration_vector = np.asarray(transformed, dtype=float)
    if len(observation) > 1:
        vehicle_mass = observation[1]
    else:
        vehicle_mass = 1.0
    if vehicle_mass == 0:
        vehicle_mass = 1.0
    limits = np.array([8000.0, 4000.0, 4000.0]) / vehicle_mass
    normalized = accelration_vector / limits
    propulsion_vector = np.zeros(4, dtype=float)
    propulsion_vector[:3] = np.clip(normalized, -1.0, 1.0)
    return propulsion_vector