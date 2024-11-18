# DS223-hw2

## Overview

This project explores Epsilon-Greedy and Thompson Sampling algorithms, which are used to solve a multi-armed bandit problem. The goal is to design an experiment that compares the cumulative rewards and regrets of both algorithms, using four advertisement options (bandits) with different rewards.

## Implementation components

Bandit Class: Provides the structure for both algorithms to interact with the environment.

Epsilon Greedy Method: Selects actions based on a decaying epsilon value, balancing exploration and exploitation.

Thompson Sampling Method: Selects actions based on sampling from the posterior distribution of rewards, focusing on the most promising actions.

## Conclusions

Based on the outputs, Epsilon Greedy outperformed Thompson Sampling in terms of cumulative reward. In addition, Thompson sampling resulted in a higher cumulative regret.

## Suggestions?

 Variance can be used for the reward to see how much it varies across multiple experiments

