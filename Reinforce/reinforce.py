from Reinforce.utilities import *
from Reinforce.policies import *

class Model:
    def __init__(self, policy, batch_size, num_envs, observation_space, action_space, learning_rate):
        self.session = tf.Session()
        self.actions = tf.placeholder(tf.uint8, [batch_size * num_envs], name="action")
        self.rewards = tf.placeholder(tf.float32, [batch_size * num_envs], name="rewards")

        self.model_predict = policy(num_envs, observation_space, action_space, reuse=False)
        self.model_train = policy(batch_size * num_envs, observation_space, action_space, reuse=True)

        action_mask = tf.one_hot(self.actions, action_space)

        self.learning_rate = tf.Variable(trainable=False, initial_value=learning_rate)
        self.loss = tf.reduce_mean(
            tf.reduce_sum(action_mask * tf.log(self.model_train.policy + 1e-13), 1) * -self.rewards)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, inputs, rewards, actions):
        self.session.run(self.optimizer, feed_dict={self.model_train.inputs: inputs,
                                                    self.rewards: rewards,
                                                    self.actions: actions})

    def update_hyperparams(self, hyperparams):
        self.session.run(self.learning_rate.assign(hyperparams))

    def hyperparams_value(self):
        return self.session.run(self.learning_rate)

    def explore(self):
        # each hyperparameter independently is randomly perturbed by a factor of 1.2 or 0.8
        factors = [0.8, 1.2]
        learning_rate = self.hyperparams_value()
        self.update_hyperparams((learning_rate * random.choice(factors)))

    def save(self, id):
        saver = tf.train.Saver()
        saver.save(self.session, "Saver/model_{}.ckpt".format(id), write_meta_graph=False)

    def load(self, id):
        saver = tf.train.Saver()
        saver.restore(self.session, "Saver/model_{}.ckpt".format(id))


    def loss_value(self, inputs, rewards, actions):
        return self.session.run(self.loss, feed_dict={self.model_train.inputs: inputs,
                                                        self.rewards: rewards,
                                                        self.actions: actions})


    def predict_action(self, inputs):
        policy = self.session.run(self.model_predict.policy, feed_dict={self.model_predict.inputs: inputs})
        actions = []
        for p in policy:
            action = action_with_policy(p)
            actions.append(action)
        return actions