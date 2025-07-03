import gymnasium as gym

# --- 环境和智能体的定义 (这部分你的代码是正确的，无需修改) ---
env = gym.make('MountainCar-v0', render_mode='human')  # 修改点：建议在这里直接加入render_mode
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
                                  env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))


class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


agent = SimpleAgent(env)


# --- play 函数 (这是主要修改区域) ---
def play(env, agent, render=False, train=False):
    episode_reward = 0.

    # 修改1: reset现在返回两个值, 并且在这里设置随机种子
    observation, info = env.reset(seed=3)

    while True:
        # render() 在新版中不再需要单独调用，因为在make中已经指定了render_mode
        # 如果你没有在 make 中指定 render_mode，才需要在这里调用 env.render()
        action = agent.decide(observation)

        # 修改2: step现在返回五个值
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        if train:
            # 注意：如果需要训练，这里的参数也需要更新
            agent.learn(observation, action, reward, terminated, truncated, info)

        # 修改3: 回合结束的判断条件
        if terminated or truncated:
            break

        observation = next_observation
    return episode_reward


# --- 主程序执行部分 ---
# env.seed(3) # 修改4: 这行代码应被移除, seed在reset中设置
episode_reward = play(env, agent)  # render参数在make中已设置
print('回合奖励 = {}'.format(episode_reward))
env.close()