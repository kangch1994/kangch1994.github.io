import matplotlib.pyplot as plt
import numpy as np

max_steps = 10000
max_runs = 2000


class Policy(object):
    def __init__(self, mode='non-constant', epsilon=0.1, action_space=10, alpha=0.1):
        self.action_space = action_space
        self.q_estimate = np.array([0.] * action_space)
        self.mode = mode
        self.alpha = alpha
        self.num_selected = np.array([0] * action_space)
        self.description = 'epsilon = ' + str(epsilon)
        if mode == 'non-constant':
            self.description += ', alpha = 1/n'
        elif mode == 'constant':
            self.description += ', alpha = ' + str(alpha)
        else:
            print('Policy mode wrong!!')
            quit()
        self.epsilon = epsilon
        # self.average_reward = np.mean(self.q_estimate)

    def freshen(self):
        self.q_estimate = np.array([0.] * self.action_space)
        self.num_selected = np.array([0] * self.action_space)

    def choose_action(self):
        if (np.random.random() < self.epsilon) or (not all(self.num_selected)):  # non-greedy
            # print('epsilon case')
            a = np.random.randint(0, self.action_space)
        else:
            a = np.argmax(self.q_estimate)
        self.num_selected[a] += 1
        return a

    def update_q(self, action, reward):
        q_old = self.q_estimate[action]
        if self.mode == 'non-constant':
            self.alpha = 1. / self.num_selected[action]
        self.q_estimate[action] = q_old + self.alpha * (reward - q_old)
        # self.average_reward = np.mean(self.q_estimate)
        # print(self.average_reward, self.q_estimate)


class VirtualEnv(object):
    def __init__(self, action_space=10):
        self.action_space = action_space
        self.values = np.random.normal(0, 1, action_space)
        self.optimal_action = np.argmax(self.values)
        # print(self.values, self.optimal_action)

    def freshen(self):
        self.values = np.random.normal(0, 1, self.action_space)
        self.optimal_action = np.argmax(self.values)

    def fluctuate(self):
        disturb = np.random.normal(0, 0.01, self.action_space)
        self.values += disturb
        self.optimal_action = np.argmax(self.values)

    def feedback(self, action):
        return np.random.normal(self.values[action], 1)


average_reward = {}
optimal_action_rates = {}


def main():
    # policies = [Policy(epsilon=0), Policy(epsilon=0.01), Policy(epsilon=0.1)]
    policies = [Policy(mode='non-constant'), Policy(mode='constant')]
    num_optimal = np.array([[0] * max_steps] * len(policies))
    reward_sum = np.array([[0.] * max_steps] * len(policies))
    env = VirtualEnv()
    for j in range(1, max_runs + 1):
        if j % 100 == 0:
            print('run : ', j)
        env.freshen()
        for pi in policies:
            pi.freshen()
            index = policies.index(pi)
            for i in range(max_steps):
                action = pi.choose_action()
                reward = env.feedback(action)
                # env.fluctuate()
                pi.update_q(action, reward)
                if action == env.optimal_action:
                    num_optimal[index][i] += 1
                reward_sum[index][i] += reward
    for index in range(len(policies)):
        pi = policies[index]
        average_reward[pi.description] = 1. * reward_sum[index] / max_runs
        optimal_action_rates[pi.description] = 1. * num_optimal[index] / max_runs


if __name__ == '__main__':
    main()
    # print(average_reward)
    plt.subplot(211)
    for p in average_reward:
        plt.plot(average_reward[p], label=p)
    plt.xlabel('Steps')
    # plt.axis([1, max_steps, 0, 1.6])
    plt.ylabel('Average reward')
    plt.legend()
    plt.subplot(212)
    for p in optimal_action_rates:
        plt.plot(optimal_action_rates[p], label=p)
    plt.xlabel('Steps')
    # plt.axis([1, max_steps, 0, 1.5])
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.show()
