from kspdg.pe1.e1_envs import PE1_E1_I1_Env
import krpc

agent = PE1_E1_I1_Env()
agent.reset()
conn = agent.conn

vessel = conn.space_center.active_vessel
vessel.control.sas = True
vessel.control.sas_mode = conn.space_center.SASMode.stability_assist