# Implementation of Augmented Random Search

###Code Authors: Vinit Sarode & Abhimanyu

# Use Code:
> python train.py

1. First element of variable "self.size" should be equal to number of possible actions. (4 for LunarLander-v0)]
2. If all rewards are equal in magnitude (reward_p & reward_n) then, sigmaR becomes zero. In such case sigmaR is set to 1 to avoid NaN. Training gets stopped as this condition is met.

# To test:
> python test.py

# Generate reults:
> python test_trials.py