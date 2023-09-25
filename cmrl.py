import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class MemoryBuffer:
    def __init__(self, mem_size, batch_size):
        self.MEMORY_SIZE = mem_size # Size of replay buffer
        self.BATCH_SIZE = batch_size # Batch size when sampling from replay buffer
        self.memory = deque(maxlen = self.MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done, ipl):
        state = np.reshape(state, [1, ipl])
        next_state = np.reshape(next_state, [1, ipl])
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        if len(self.memory) < self.BATCH_SIZE: return []
        else: return random.sample(self.memory, self.BATCH_SIZE)

class CESAgent:
    def __init__(self, env, env_seed, noise_table_size, hl1, hl2, sigma, lamb, mu, mem_buff):
        self.env = env
        self.env_seed = env_seed
        self.noise_table = np.random.randn(noise_table_size) # Noise table
        self.memory = mem_buff # Memory Buffer

        # NN's layer size
        self.ipl = self.env.observation_space.shape[0]
        self.hl1 = hl1
        self.hl2 = hl2
        self.opl = self.env.action_space.n

        # CES's hyper params
        self.SIGMA = sigma # Gaussian noise scalar
        self.LAMB = lamb # Population size
        self.MU = mu # Parent size
        self.theta = np.random.randn(self.ipl * self.hl1 + self.hl1 * self.hl2 + self.hl2 * self.opl) # Model weight
        self.w = np.full([self.MU], 1 / self.MU) # Update weight
    
    # Sample a single segment of the noise table, return a single noise signal
    def _sample_noise(self):
        noise_ind = np.random.randint(self.noise_table.shape[0] - self.theta.shape[0])
        return self.noise_table[noise_ind:noise_ind + self.theta.shape[0]]

    # Sample LAMB noise signals from the noise table
    def sample_noise(self):
        noises = np.empty((self.LAMB, self.theta.shape[0]))
        for i in range(self.LAMB):
            noises[i] = self._sample_noise()
        return noises
    
    # NN's feed forward
    def get_action(self, state, _theta):
        theta0 = _theta[:self.ipl * self.hl1].reshape((self.ipl, self.hl1))
        theta1 = _theta[self.ipl * self.hl1:self.ipl * self.hl1 + self.hl1 * self.hl2].reshape((self.hl1, self.hl2))
        theta2 = _theta[self.ipl * self.hl1 + self.hl1 * self.hl2:].reshape((self.hl2, self.opl))

        out0 = np.dot(state, theta0)
        out1 = np.dot(out0, theta1)
        out2 = np.dot(out1, theta2)

        return np.argmax(out2)
    
    # Run through an RL episode
    def rollout(self, _theta, render = False):
        self.env.seed(self.env_seed)
        state = self.env.reset()
        total_reward = 0
        while True:
            if render: self.env.render()
            action = self.get_action(state, _theta)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.remember(state, action, reward, next_state, done, self.ipl) # Collect TD frame
            total_reward += reward
            if done: break
            state = next_state
        return total_reward
    
    # CES's main component
    def explore(self):
        fitness = np.empty(self.LAMB)
        noise = self.sample_noise()

        # Create self.LAMB number of clones and run each of them through env
        for i in range(self.LAMB):
            _theta = self.theta + self.SIGMA * noise[i]
            fitness[i] = self.rollout(_theta)
        
        # Get indices of MU solutions with best fitness
        ind = np.argpartition(fitness, -self.MU)[-self.MU:]
        ind = ind[np.argsort(fitness[ind])]

        # Update weight
        for i in range(self.MU):
            noise_ind = ind[i]
            self.theta += self.SIGMA * self.w[i] * noise[noise_ind]
    
    # Utility methods
    def set_weights(self, _theta): self.theta = _theta
    def get_weights(self): return self.theta
    def save_model(self): np.savetxt("CES-" + self.env.unwrapped.spec.id + "-seed-" + str(self.env_seed), self.get_weights())
    def load_model(self, file_name): self.set_weights(np.loadtxt(file_name))
    def test(self): self.rollout(self.get_weights(), render = True)

# Compare to the original DQN model, the DQN in Memetic-RL framework
# only does the job of local exploitation using experience replay
class DQNAgent:
    def __init__(self, ipl, hl1, hl2, opl, gamma, lr):
        self.GAMMA = gamma # Discount factor
        self.LEARNING_RATE = lr # For updating agent when using experience replay
        
        # NN's layers
        self.ipl = ipl
        self.hl1 = hl1
        self.hl2 = hl2
        self.opl = opl

        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential()
        model.add(Dense(self.hl1, input_shape=(self.ipl,), activation="relu"))
        model.add(Dense(self.hl2, activation="relu"))
        model.add(Dense(self.opl, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

        return model

    def experience_replay(self, batch):
        for state, action, reward, next_state, done in batch:
            if not done:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(next_state)[0]))
            else:
                q_update = reward
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
    
    # Update weights for both current and target model
    def set_weights(self, theta):
        theta0 = theta[:self.ipl * self.hl1].reshape((self.ipl, self.hl1))
        theta1 = theta[self.ipl * self.hl1:self.ipl * self.hl1 + self.hl1* self.hl2].reshape((self.hl1, self.hl2))
        theta2 = theta[self.ipl * self.hl1 + self.hl1 * self.hl2:].reshape((self.hl2, self.opl))

        layer0 = self.model.layers[0].get_weights()
        layer0[0] = theta0
        layer1 = self.model.layers[1].get_weights()
        layer1[0] = theta1
        layer2 = self.model.layers[2].get_weights()
        layer2[0] = theta2
        
        self.model.layers[0].set_weights(layer0)
        self.model.layers[1].set_weights(layer1)
        self.model.layers[2].set_weights(layer2)
    
    def get_weights(self):
        theta0 = self.model.layers[0].get_weights()[0].reshape((self.ipl * self.hl1, ))
        theta1 = self.model.layers[1].get_weights()[0].reshape((self.hl1 * self.hl2, ))
        theta2 = self.model.layers[2].get_weights()[0].reshape((self.hl2 * self.opl, ))

        return np.concatenate((theta0, theta1, theta2))

class MRLAgent:
    def __init__(self, env_name, env_seed = np.random.randint(1000), noise_table_size = 1000000, hl1 = 48, hl2 = 48, sigma = 0.5, lamb = 50, mu = 10, gamma = 0.95, lr = 0.00005, memory_size = 100000, batch_size = 32):
        env = gym.make(env_name)
        self.dqn_agent = DQNAgent(env.observation_space.shape[0], hl1, hl2, env.action_space.n, gamma, lr)
        self.es_agent = CESAgent(env, env_seed, noise_table_size, hl1, hl2, sigma, lamb, mu, MemoryBuffer(memory_size, batch_size))

    # Train for a fix number of run
    def train(self, num_run):
        rewards = np.empty(num_run)
        for run in range(1, num_run + 1):
            # Global exploration, accumulate TD frame
            self.es_agent.explore()

            # Local exploitation
            self.dqn_agent.set_weights(self.es_agent.get_weights())
            self.dqn_agent.experience_replay(self.es_agent.memory.sample())
            self.es_agent.set_weights(self.dqn_agent.get_weights())

            # Rollout
            reward = self.es_agent.rollout(self.es_agent.get_weights())
            rewards[run - 1] = reward
            print("Run: {}, reward: {}".format(run, reward))
        # Save rewards for plotting
        np.savetxt("MRL-{}-seed-{}-rewards".format(self.es_agent.env.unwrapped.spec.id, self.es_agent.env_seed), rewards)       

    # Utility methods
    def get_weights(self): return self.es_agent.get_weights()
    def set_weights(self, theta): self.es_agent.set_weights(theta)
    def save_model(self): np.savetxt("MRL-{}-seed-{}-weights".format(self.es_agent.env.unwrapped.spec.id, self.es_agent.env_seed), self.get_weights())
    def load_model(self, file_name): self.set_weights(np.loadtxt(file_name))
    def test(self): self.es_agent.rollout(self.es_agent.get_weights(), render = True)
