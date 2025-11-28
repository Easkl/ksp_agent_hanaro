from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.pe1.e1_envs import PE1_E1_I1_Env
from kspdg.agent_api.runner import AgentEnvRunner
import time

class PulsedAgent(KSPDGBaseAgent):
    def __init__(self):
        super().__init__()
        self.is_on = False

    def get_action(self, observation):
        time.sleep(1.0)
        self.is_on = not self.is_on

        if self.is_on:
            return {
                "burn_vec": [0, 1.0, 0, 0.3],
                "ref_frame": 0
            }
        else:
            return {
                "burn_vec": [0, 0.0, 0, 0.0],
                "ref_frame": 0
            }

if __name__ == "__main__":
    agent = PulsedAgent()
    runner = AgentEnvRunner(
        agent=agent,
        env_cls=PE1_E1_I1_Env,
        env_kwargs=None,
        runner_timeout=100,
        debug=False,
        enable_telemetry=False
    )
    runner.run()
