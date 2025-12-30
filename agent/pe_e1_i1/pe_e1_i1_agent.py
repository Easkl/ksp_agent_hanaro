from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner


class PE1_E1_I1_Env_SAS(PE1_E1_I1_Env):
    def _reset_vessels(self):
        super()._reset_vessels()
        self.vesPursue.control.sas_mode = self.conn.space_center.SASMode.stability_assist

class PE_E1_I1_Agent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        return {
            "burn_vec": [1.0, 0, 0, 1.0],
            "ref_frame": 0,
            "vec_type": 0,
            }
    
if __name__ == "__main__":
    pe_e1_i1_agent = PE_E1_I1_Agent()    
    runner = AgentEnvRunner(
        agent=pe_e1_i1_agent, 
        env_cls=PE1_E1_I1_Env_SAS, 
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
        enable_telemetry=True)
    print(runner.run())
    