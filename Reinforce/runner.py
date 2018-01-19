from Reinforce.utilities import discount

class Runner():

    def __init__(self, env, discount_rate, num_episodes, batch_size):
        self.env = env
        self.discount_rate = discount_rate
        self.observation = env.reset()

        self.episodic_rewards = []
        self.episodic_performance = []
        self.average_episodic_performance = 0
        self.episode = 0

        self.num_episodes = num_episodes
        self.batch_size = batch_size

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

            #if (self.episode % 100) == 0:
            #    print("Episode {}: {}".format(self.episode, self.performance))
            self.episode += 1

        if len(self.episodic_performance) == self.num_episodes:
            self.average_episodic_performance = sum(self.episodic_performance)/self.num_episodes
            self.episodic_performance.pop(0)

    def step(self, model):
        batch_observations = []
        batch_rewards = []
        batch_actions = []
        batch_dones = []

        while True:
            action = model.predict_action([self.observation])[0]
            batch_observations.append(self.observation)
            self.observation, reward, done = self.env_step(action)
            self.evaluate(reward, done)
            batch_rewards.append(reward)
            batch_actions.append(action)
            batch_dones.append(done)

            if len(batch_observations) == self.batch_size:
                discounted_rewards = discount(batch_rewards, batch_dones, self.discount_rate)
                return batch_observations, discounted_rewards, batch_actions