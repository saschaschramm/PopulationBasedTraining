import tensorflow as tf

class Model():
    def __init__(self):
        tf.set_random_seed(0)
        self.steps = 0
        self.session = tf.Session()
        self.learning_rate = 0.03
        self.hyperparams = tf.Variable(trainable=False, initial_value=[0.0,0.0])
        self.params = tf.get_variable("params", 2)
        self.loss = -(1.2 - (self.hyperparams[0] * tf.square(self.params[0]) +
                             self.hyperparams[1] * tf.square(self.params[1])))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run(session=self.session)

    def update_params(self, params):
        self.session.run(self.params.assign(params))

    def update_hyperparams(self, hyperparams):
        self.session.run(self.hyperparams.assign(hyperparams))

    def eval(self):
        return self.session.run(self.loss)

    def get_params(self):
        return self.session.run(self.params)

    def get_hyperparams(self):
        return self.session.run(self.hyperparams)

    def train(self):
        self.steps += 1
        self.session.run([self.loss, self.optimizer])