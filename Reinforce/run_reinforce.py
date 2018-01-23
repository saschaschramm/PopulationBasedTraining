from Reinforce.Environment.catch import Catch
from Reinforce.policies import PolicyFullyConnected
from Reinforce.PopulationBasedTraining.manager import Manager
from Summary import summary
from Reinforce.reinforce import Model
from Reinforce.runner import Runner

def init_env(seed):
    return Catch(5)

def run():

    model_params = {
        "policy": PolicyFullyConnected,
        "num_envs": 1,
        "observation_space": [5, 5],
        "action_space": 8,
        "learning_rate": 0.01,
        "discount_rate": 0.99,
        "batch_size": 10
    }

    runner_params = {
        "num_workers": 5,
        "ready_steps": 20000, # Number of steps until model is ready.
        "performance_num_episodes": 10,  # Number of episodes to measure model's average performance.
        "save_summary_steps": 1000,
        "summary_log_dir": "Test",
    }

    manager = Manager(init_env=init_env,
                      runner=Runner,
                      runner_params=runner_params,
                      model=Model,
                      model_params=model_params
                      )

    time_steps = 6
    for i in range(0, time_steps):
        manager.step()
        # worker are synchronized

    file_reader = summary.FileReader("Test")
    file_reader.plot(True)


def main():
    run()

if __name__ == '__main__':
   main()