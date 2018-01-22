from Reinforce.utilities import discount
from Summary import summary

class Runner():

    def __init__(self, env, model, model_params, runner_params, worker_id):
        self.env = env
        self.model = model
        self.worker_id = worker_id
        self.discount_rate = model_params["discount_rate"]
        self.observation = env.reset()

        self.episodic_rewards = []
        self.episodic_performance = []
        self.average_episodic_performance = 0
        self.episode = 0

        self.performance_num_episode = runner_params["performance_num_episode"]
        self.batch_size = model_params["batch_size"]
        self.save_summary_steps = runner_params["save_summary_steps"]

        self.file_writer = summary.FileWriter(runner_params["summary_log_dir"])

    def env_step(self, action):
        observation, reward, done, _ = self.env.step(action)
        if done:
            observation = self.env.reset()
        return observation, reward, done

    def evaluate(self, reward, done):
        self.episodic_rewards.append(reward)
        if done:
            self.episodic_performance.append(sum(self.episodic_rewards) / len(self.episodic_rewards))
            self.episodic_rewards = []

            if (self.episode % self.save_summary_steps) == 0 and (self.episode != 0):
                #print("{}   {}".format(self.episode, self.average_episodic_performance))
                self.file_writer.add_summary(self.worker_id, self.episode, self.average_episodic_performance)
            self.episode += 1

        if len(self.episodic_performance) == self.performance_num_episode:
            self.average_episodic_performance = sum(self.episodic_performance)/self.performance_num_episode
            self.episodic_performance.pop(0)

    def step(self):
        batch_observations = []
        batch_rewards = []
        batch_actions = []
        batch_dones = []

        while True:
            action = self.model.predict_action([self.observation])[0]
            batch_observations.append(self.observation)
            self.observation, reward, done = self.env_step(action)
            self.evaluate(reward, done)
            batch_rewards.append(reward)
            batch_actions.append(action)
            batch_dones.append(done)

            if len(batch_observations) == self.batch_size:
                discounted_rewards = discount(batch_rewards, batch_dones, self.discount_rate)
                return batch_observations, discounted_rewards, batch_actions