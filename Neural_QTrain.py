import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.9 # discount factor
INITIAL_EPSILON =  0.9# starting value of epsilon
FINAL_EPSILON =  20# final value of epsilon
EPSILON_DECAY_STEPS =  0.5# decay period

HIDDEN_NODES = 100

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
# state_in: takes the current state of the environment, which is 
#       represented in our case as a sequence of reals.
# action_in: accepts a one-hot action input. It should be used to "mask"
#       the q-values output tensor and return a q-value for that action.
# target_in: is the Q-value we want to move the network towards producing.
#       Note that this target value is not fixed - this is one of the
#       components that seperates RL from other forms of machine learning.
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph

# first layer
w1 = tf.get_variable('w1', shape=[STATE_DIM, HIDDEN_NODES])
b1 = tf.get_variable('b1', shape=[1, HIDDEN_NODES],\
        initializer=tf.constant_initializer(0.0))

# second layer
w2 = tf.get_variable('w2', shape=[HIDDEN_NODES, HIDDEN_NODES])
b2 = tf.get_variable('b2', shape=[1, HIDDEN_NODES],\
        initializer=tf.constant_initializer(0.0))

# third layer
w2 = tf.get_variable('w3', shape=[HIDDEN_NODES, HIDDEN_NODES])
b2 = tf.get_variable('b3', shape=[1, HIDDEN_NODES],\
        initializer=tf.constant_initializer(0.0))

# calculation
output_1 = tf.tanh(tf.matmul(state_in, w1) + b1)
output_2 = tf.tanh(tf.matmul(output_1, w2) + b2)
output_3 = tf.tanh(tf.matmul(output_2, w3) + b3)

# TODO: Network outputs
# q_values: Tensor containing Q-values for all available actions i.e.
#       if the action space is 8 this will be a rank-1 tensor of length 8
# q_action: This should be a rank-1 tensor containing 1 element.
#       This value should be the q-value for the action set in the
#       action_in placeholder
# Loss/Optimizer Definition You can define any loss function you feel is
#       appropriate. Hint: should be a function of target_in and
#       q_action. You should also make careful choice of the optimizer
#       to use. 
q_values = output_3
q_action = tf.reduce_sum(tf.multiply(q_values, action_in),\
        reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss = tf.reduce_sum(tf.square(target_in - q_action))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target =

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
