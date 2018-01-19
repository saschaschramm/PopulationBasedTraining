import tensorflow as tf
import random
import numpy as np

def global_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def action_with_policy(policy):
    rand = random.uniform(0, 1)
    cumulated_sum = np.cumsum(policy)
    for i in range(0, len(cumulated_sum)):
        if rand <= cumulated_sum[i]:
            return i
    return 0

def discount(rewards, dones, discount_rate):
    discounted = []
    total_return = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            total_return = reward
        else:
            total_return = reward + discount_rate * total_return
        discounted.append(total_return)
    return np.asarray(discounted[::-1])