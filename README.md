# Evolutionary algos for solving RL problems
# Src

```
├── cem.py                  # Source code for simple Cross Entropy Method (pytorch version)
├── ces.py                  # Source code for Canonical Evolution Strategy
├── cmrl.py                 # Source code for Canonical Memetic Reinforcement Learning
├── cmaes.py                # Source code for Covariance Matrix Adaptation Evolution Strategy
├── cmamrl.py               # Source code for Covariance Matrix Adaptation Memetic Reinforcement Learning
├── train.py                # Main script for training
└── requirements.txt        # Python dependencies
```
# Install dependencies
In your own conda env, try this:
```
pip install -r requirements.txt
```
# Run the source code
## Training
All 4 algorithms are provided in separate files with respective name. To run each algorithm, import the appropriate *.py* file, create an *Agent* object and run via the *train* method.

A simple example of solving the *CartPole-v1* environment using the *CES* algorithm:
```
python
>>> from ces import *
>>> agent = CESAgent("CartPole-v1")
>>> agent.train(100)
>>> agent.save_model()
```

In this work, each algorithm is run 100 times with different environment seed. This can be achived using the snippet:
```
# Solving the CartPole-v1 environment using CES algorithm with 30 different seeds
python
>>> from ces import *
>>> for seed in range(30):
>>>     agent = CESAgent("CartPole-v1", env_seed = seed)
>>>     agent.train(100)
>>>     agent.save_model()
```

Using CEM to solve CartPole-v0

<img src="cem.png" div align=middle width = "55%" />

## Testing
After learning, a model weights can be saved using the *save_model* method. A learnt model can be tested using the *load_model* and *test* methods.
```
python
>>> from ces import *
>>> agent = CESAgent("CartPole-v1")
>>> agent.load_model("CES-CartPole-v1-seed-0-weights")
>>> agent.test(num_test = 3)
```
