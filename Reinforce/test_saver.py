from Reinforce.policies import PolicyFullyConnected
from Reinforce.reinforce import Model
from Reinforce.utilities import *
import unittest

class TestSaver(unittest.TestCase):

    def test_saver(self):
        global_seed(0)
        rewards = [1.0]
        observations = [[[0.1, 0.3], [0.4, 0.9]]]
        actions = [1.0]
        batch_size = 1
        policy = PolicyFullyConnected

        model = Model(policy=policy, batch_size=batch_size, num_envs=1, observation_space=[2, 2], action_space=2,
                      learning_rate=0.01)

        print("***********************")
        print("Save:")
        loss_save = model.loss_value(observations, rewards, actions)
        print("Loss", loss_save)
        model.explore()
        hyperparams = model.hyperparams_value()

        print("Hyperparameter", hyperparams)
        model.save(0)

        print("***********************")
        print("Reset:")
        model.session.run(tf.global_variables_initializer())
        loss = model.loss_value(observations, rewards, actions)
        print("Loss ", loss)
        hyperparams = model.hyperparams_value()
        print("Hyperparameter", hyperparams)

        model.load(0)
        print("***********************")
        print("Restore:")
        loss_load = model.loss_value(observations, rewards, actions)
        print("Loss ", loss_load)

        hyperparams = model.hyperparams_value()
        print("Hyperparameter ", hyperparams)

        self.assertEqual(loss_save, loss_load)


if __name__ == '__main__':
    unittest.main()
