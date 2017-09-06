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

    order_list=[[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]];
    tr_reward=0;ts_reward=0;
    for episode in range(EPISODES):
        # training with blue cube, red sphere and blue sphere
        program_order_idx=np.random.randint(1,4);
        #program_order_idx=0;
        env.set_order(program_order_idx,order_list[program_order_idx]);
        state = env.reset();
        # Train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state,env.program_order)
            next_state,reward,done,_ = env.step(action)
            tr_reward+=reward;
            agent.perceive(state,action,reward,next_state,done,env.program_order)
            state = next_state
            if done:
                break
        # Testing:
        if(episode % 100 == 0 and episode > 100):
            for i in range(TEST):
                # testing with red cube
                program_order_idx=0;
                env.set_order(program_order_idx,order_list[program_order_idx]);
                state = env.reset();
                for j in range(env.spec.timestep_limit):
                    action = agent.action(state,env.program_order);
                    state,reward,done,_ = env.step(action);
                    ts_reward += reward;
                    if done:
                        break;
            ave_tr_reward = tr_reward/100;
            ave_ts_reward = ts_reward/TEST;
            tr_reward=0;ts_reward=0;
            print("episode: "+str(episode)+", Training Average Reward: "+str(ave_tr_reward)+", Evaluation Average Reward: "+str(ave_ts_reward));

if __name__ == '__main__':
    main()
