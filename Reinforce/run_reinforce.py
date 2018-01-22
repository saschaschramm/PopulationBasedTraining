from Reinforce.Environment.catch import Catch
from Reinforce.policies import PolicyFullyConnected
from Reinforce.PopulationBasedTraining.manager import Manager
from Summary import summary

def run():
    env = Catch(5)
    model_params = {
        "policy": PolicyFullyConnected,
        "num_envs": 1,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "learning_rate": 0.01,
        "discount_rate": 0.99,
        "batch_size": 10
    }

    runner_params = {
        "num_workers": 5,
        "ready_steps": 1000, # Number of steps until model is ready.
        "performance_num_episode": 10,  # Number of episodes to measure model's average performance.
        "save_summary_steps": 100,
        "summary_log_dir": "Test",
    }

    manager = Manager(env=env,
                      runner_params = runner_params,
                      model_params=model_params
                      )

    time_steps = 10
    for i in range(0, time_steps):
        manager.step()
        # worker are synchronized

    file_reader = summary.FileReader("Test")
    file_reader.plot(True)

def main():
    run()

if __name__ == '__main__':
   main()