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

    # command: 0-not, 1-and, 2-or
    # col_shape: 0-red, 1-blue, 2-green, 3-A, 4-B, 5-box, 6-cylinder, 7-sphere, 8-hand
    order_list=[[[1,7,0],[2,8,-1]], #target1
                [[1,6,2],[2,8,-1]], #target2
                [[1,5,1],[2,8,-1]]];#target3
    for episode in range(EPISODES):
        program_order_idx=np.random.randint(3);
        env.set_order(program_order_idx,order_list[program_order_idx]);
        state = env.reset();
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state,env.program_order)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if(episode % 100 == 0 and episode > 100):
            total_reward=0;
            for i in range(TEST):
                program_order_idx=np.random.randint(3);
                env.set_order(program_order_idx,order_list[program_order_idx]);
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
