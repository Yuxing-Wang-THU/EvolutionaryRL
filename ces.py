import gym
import numpy as np
from collections import deque

class CESAgent:
    def __init__(self, env_name, seed = 0, noise_table_size = 1000000, hl1 = 48, hl2 = 48, sigma = 0.5, lamb = 50, mu = 5):
        self.env = gym.make(env_name)
        self.env.seed(seed)
        np.random.seed(seed)
        self.noise_table = np.random.randn(noise_table_size) # Noise table

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
        state = self.env.reset()
        total_reward = 0
        while True:
            if render: self.env.render()
            action = self.get_action(state, _theta)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done: break
            state = next_state
        return total_reward
    
    # CES main loop
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
    
    # Train for a fix number of run
    def train(self, num_run):
        rewards = []
        for run in range(1, num_run + 1):
            self.explore()
            reward = self.rollout(self.get_weights())
            rewards.append(reward)
            print("Run: {}, Current reward {}, Average reward: {}".format(run, reward, np.average(rewards)))
        # Save rewards for plotting
        np.savetxt("CES-" + self.env.unwrapped.spec.id + "-seed-" + str(self.env_seed) + "-rewards", rewards)
    
    # Utility methods
    def set_weights(self, _theta): self.theta = _theta
    def get_weights(self): return self.theta
    def save_model(self): np.savetxt("CES-" + self.env.unwrapped.spec.id + "-seed-" + str(self.env_seed) + "-weights", self.get_weights())
    def load_model(self, file_name): self.set_weights(np.loadtxt(file_name))
    def test(self, num_test = 1, verbose = True, render = True, random_seed = False): 
        rewards = np.empty(num_test)
        for i in range(num_test):
            if random_seed: self.env_seed = np.random.randint(1000)
            reward = self.rollout(self.get_weights(), render = render)
            rewards[i] = reward
            if verbose: print("Reward: {}".format(reward))
        self.env.close()
        return np.mean(rewards)
        if verbose: print("Average reward over {} test runs: {}".format(num_test, np.mean(rewards)))