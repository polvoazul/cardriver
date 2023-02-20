from cardriver.env import SimpleRoad


def show_test_run(model, env=None):
    if not env:
        env = SimpleRoad()
        observation = env.reset()
    else:
        observation = env._get_obs()
    total_reward = 0
    env.render()
    print("Starting!")
    for t in range(100):
        state = None
        action, state, info = model(observation, state)
        observation, reward, done, info = env.step(action)
        total_reward += reward    
        print(f'Reward: {total_reward:0.2f}')
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(f'FINAL Reward: {total_reward:0.2f}')
            break
# algo.compute_single_action(observation, full_fetch=True, state=state)

def show_hard_test_run(model):
    hard_env = SimpleRoad()
    hard_env.light.init_time = 7 # Start with RED
    hard_env.speed = 5
    print('HARD TEST')
    show_test_run(model, env=hard_env)
