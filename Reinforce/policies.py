import tensorflow as tf

class PolicyFullyConnected():

    def __init__(self, batch_size, observation_space, action_space, reuse=False):
        height, width = observation_space
        self.inputs = tf.placeholder(tf.float32, (batch_size, height, width))
        with tf.variable_scope("model", reuse=reuse):
            inputs_reshaped = tf.reshape(self.inputs, [batch_size, width * height])
            hidden = tf.layers.dense(inputs=inputs_reshaped, units=200, activation=tf.nn.relu)
            logits_policy = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        self.policy = tf.nn.softmax(logits_policy)