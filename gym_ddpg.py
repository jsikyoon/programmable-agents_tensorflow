import filter_env
from ddpg import *
import gc
gc.enable()

ENV_NAME = 'PA-v1'
EPISODES = 100000
TEST = 10

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)

    for episode in range(EPISODES):
        program_order=np.random.randint(3);
        env.set_order(0);
        state = env.reset();
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if(episode % 100 == 0 and episode > 100):
            total_reward=0;
            for i in range(TEST):
                state = env.reset();
                for j in range(env.spec.timestep_limit):
                    action = agent.action(state);
                    state,reward,done,_ = env.step(action);
                    total_reward += reward;
                    if done:
                        break;
            ave_reward = total_reward/TEST;
            print("episode: "+str(episode)+", Evaluation Average Reward: "+str(ave_reward));

if __name__ == '__main__':
    main()
