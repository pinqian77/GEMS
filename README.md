# GEMS - Goal-aware Exploration with Multi-level Supervision 
This is the implementation for the paper 'Towards the oracle: a goal-aware exploration mechanism for reinforcement learning based portfolio optimization'.

### Run Environment
- `python`: 3.6.12
- `PyTorch`: 1.12.1
- `gym`: 0.17.2

### Overview Framework

The proposed method is mainly composed of three parts. The oracle module is trained by data containing future information and learns an optimal policy to provide instruction to the student module. The student module is trained under the guidance of the oracle and is composed of two-layer controllers. The higher-level controller is for goal generation and the lower-level controller is for trade execution. The third module provides a goal-aware exploration mechanism, which allows the agent to perform actions rationally following the goal. Note that the oracle module only provides guidance during the training session. During the testing phase, the decision-making of the action is done by the student alone.

### Results
Each algorithm is trained for 200 episodes. Each episode selects an 1000-step consecutive holding period at random. 

Further interpretation work will be presented in the paper.