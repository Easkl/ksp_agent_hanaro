'''이 파일에서 해야 하는것 navigation.py의 navigation함수에서 반환한 가속도 벡터를
받아서 rcs의 분사 벡터로 전환해서 크기 3의 array를 반환해야 한다'''
import numpy as np
from navigation import navigation
import pe_e1_i1_agent

accelration_vector = navigation(pe_e1_i1_agent.observation)


def propulsion(accelration_vector):
    propulsion_vector = np.zeros(4)
    if accelration_vector[0] >= 8000 / pe_e1_i1_agent.observation[1]:  # 만약 연료가 8톤 이상 남아있다면
        propulsion_vector[0] = 1.0
    else:
        propulsion_vector[0] = accelration_vector[0] / (8000 / pe_e1_i1_agent.observation[1])
    if accelration_vector[1] >= 4000 / pe_e1_i1_agent.observation[1]:
        propulsion_vector[1] = 1.0
    else:
        propulsion_vector[1] = accelration_vector[1] / (4000 / pe_e1_i1_agent.observation[1])
    if accelration_vector[2] >= 4000 / pe_e1_i1_agent.observation[1]:
        propulsion_vector[2] = 1.0
    else:
        propulsion_vector[2] = accelration_vector[2] / (4000 / pe_e1_i1_agent.observation[1])
    return propulsion_vector
