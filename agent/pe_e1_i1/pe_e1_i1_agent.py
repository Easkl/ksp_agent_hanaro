from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner
from observation_log import append_observation, reset_history
from analysis_log import reset_analysis
import propulsion
import navigation
import time

class PulsedAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        reset_history()
        reset_analysis()
        self.is_on = True

    def get_action(self, observation):
        time.sleep(1.0)
        append_observation(observation)
        if self.is_on == True:
            self.is_on = not self.is_on
            accel_command = navigation.navigation(observation)
            return {
            "burn_vec": propulsion.propulsion(accel_command),
            "ref_frame": 0
            }
        if self.is_on == False:
            self.is_on = not self.is_on
            return {
            "burn_vec": [0.0, 0.0, 0.0, 0.0],
            "ref_frame": 0
            }
        
if __name__ == "__main__":
    agent = PulsedAgent()
    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PE1_E1_I1_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=True,
        enable_telemetry=True
    )
    runner.run()
