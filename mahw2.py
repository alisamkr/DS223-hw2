import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Bandit_Reward = [1, 2, 3, 4]  

# Abstract Bandit class
class Bandit:
    def __init__(self, reward_probabilities):
        self.reward_probabilities = reward_probabilities

    def pull(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError

#Epsilon-Greedy class
class EpsilonGreedy(Bandit):
    def __init__(self, reward_probabilities, epsilon=0.1):
        super().__init__(reward_probabilities)
        self.epsilon = epsilon
        self.q_values = [0] * len(reward_probabilities)  # estimated values for each action
        self.action_counts = [0] * len(reward_probabilities)

    def pull(self):
        if random.random() < self.epsilon:
            action = random.choice(range(len(self.reward_probabilities)))  # exploration
        else:
            action = np.argmax(self.q_values)  # exploitation
        reward = random.gauss(self.reward_probabilities[action], 1)  # Gaussian noise
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]  # Update Q-value
        return reward

#Thompson Sampling 
class ThompsonSampling(Bandit):
    def __init__(self, reward_probabilities, alpha=1, beta=1):
        super().__init__(reward_probabilities)
        self.alpha = [alpha] * len(reward_probabilities)
        self.beta = [beta] * len(reward_probabilities)

    def pull(self):
        sampled_theta = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.reward_probabilities))]
        action = np.argmax(sampled_theta)
        reward = random.gauss(self.reward_probabilities[action], 1) 
        if reward > self.reward_probabilities[action]:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
        return reward

def run_experiment(bandit_class, reward_probabilities, num_trials=20000, epsilon=0.1, alpha=1, beta=1):
    if bandit_class == EpsilonGreedy:
        bandit = bandit_class(reward_probabilities, epsilon)
    else:
        bandit = bandit_class(reward_probabilities, alpha, beta)
    
    rewards = []
    cumulative_rewards = []
    cumulative_regret = []
    optimal_action = np.argmax(reward_probabilities)
    total_reward = 0
    total_regret = 0
    
    for t in range(1, num_trials + 1):
        reward = bandit.pull()
        total_reward += reward
        total_regret += reward_probabilities[optimal_action] - reward
        rewards.append(reward)
        cumulative_rewards.append(total_reward)
        cumulative_regret.append(total_regret)
        
        if bandit_class == EpsilonGreedy:
            bandit.epsilon = epsilon / np.sqrt(t)  # Decaying epsilon
        
    return rewards, cumulative_rewards, cumulative_regret

#visualizations
def plot_results(cumulative_rewards_eg, cumulative_rewards_ts, cumulative_regret_eg, cumulative_regret_ts):
    plt.figure(figsize=(12, 6))
    
    #cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_rewards_eg, label='Epsilon Greedy')
    plt.plot(cumulative_rewards_ts, label='Thompson Sampling')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards')
    plt.legend()

    #cumulative regret
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_regret_eg, label='Epsilon Greedy')
    plt.plot(cumulative_regret_ts, label='Thompson Sampling')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def save_results_to_csv(rewards_eg, rewards_ts, algorithm_names):
    data = []
    for i in range(len(rewards_eg)):
        data.append([Bandit_Reward[i % len(Bandit_Reward)], rewards_eg[i] if i < len(rewards_eg) else rewards_ts[i], algorithm_names[i % 2]])
    df = pd.DataFrame(data, columns=["Bandit", "Reward", "Algorithm"])
    df.to_csv("bandit_rewards.csv", index=False)

def main():
    num_trials = 20000
    epsilon = 0.1
    alpha = 1
    beta = 1

#experiment for both 
    rewards_eg, cumulative_rewards_eg, cumulative_regret_eg = run_experiment(EpsilonGreedy, Bandit_Reward, num_trials, epsilon)
    rewards_ts, cumulative_rewards_ts, cumulative_regret_ts = run_experiment(ThompsonSampling, Bandit_Reward, num_trials, alpha, beta)

    plot_results(cumulative_rewards_eg, cumulative_rewards_ts, cumulative_regret_eg, cumulative_regret_ts)
    
#save aa CSV
    save_results_to_csv(rewards_eg, rewards_ts, ['Epsilon Greedy'] * num_trials + ['Thompson Sampling'] * num_trials)

    print("Final Cumulative Reward (Epsilon Greedy):", cumulative_rewards_eg[-1])
    print("Final Cumulative Reward (Thompson Sampling):", cumulative_rewards_ts[-1])
    print("Final Cumulative Regret (Epsilon Greedy):", cumulative_regret_eg[-1])
    print("Final Cumulative Regret (Thompson Sampling):", cumulative_regret_ts[-1])

if __name__ == "__main__":
    main()
    