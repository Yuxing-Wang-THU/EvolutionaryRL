import cma
import gym
import numpy as np

class CMAESAgent:
    def __init__(self, env_name, env_seed = 0, hl1 = 48, hl2 = 48, sigma = 0.5, lamb = 50, mu = 10):
        self.env = gym.make(env_name)
        self.env_seed = env_seed

        # NN's layer size
        self.ipl = self.env.observation_space.shape[0]
        self.hl1 = hl1
        self.hl2 = hl2
        self.opl = self.env.action_space.n

        # MRL's hyper params
        self.theta = np.random.randn(self.ipl * self.hl1 + self.hl1 * self.hl2 + self.hl2 * self.opl) # Model weight
        self.es = cma.CMAEvolutionStrategy(self.theta, sigma0 = sigma, inopts = {"popsize": lamb, "CMA_mu": mu}) # External CMA-ES module from original paper
    
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
            total_reward += reward
            if done: break
            state = next_state
        return total_reward
    
    # CMAES's main loop
    def explore(self):
        solutions = self.es.ask(xmean = self.get_weights()) # Sample candidate solutions from Covariance Matrix with Mean is our current model weights
        fitness = [-self.rollout(solution) for solution in solutions] # We use negative value of the objective function since we are trying to maximize it instead of minimize
        self.es.tell(solutions, fitness) # Update the Mean and Covariance Matrix of the next generation
        self.set_weights(self.es.result.xfavorite) # The Mean of the next generation will be consider our new model weight
    
    # Train for a fix number of run
    def train(self, num_run):
        rewards = np.empty(num_run)
        for run in range(1, num_run + 1):
            self.explore() # Same interface with CES's source code for convenience
            reward = self.rollout(self.get_weights())
            rewards[run - 1] = reward
            print("Run: {}, reward: {}".format(run, reward))
        # Save rewards for plotting
        np.savetxt("CMAES-" + self.env.unwrapped.spec.id + "-seed-" + str(self.env_seed) + "-rewards", rewards)
    
    # Utility methods
    def set_weights(self, _theta): self.theta = _theta
    def get_weights(self): return self.theta
    def save_model(self): np.savetxt("CMAES-" + self.env.unwrapped.spec.id + "-seed-" + str(self.env_seed) + "-weights", self.get_weights())
    def load_model(self, file_name): self.set_weights(np.loadtxt(file_name))
    def test(self): self.rollout(self.get_weights(), render = True)
