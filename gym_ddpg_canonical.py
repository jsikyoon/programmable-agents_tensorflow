import filter_env_canonical
from gym import wrappers
from ddpg_canonical import *
import gc
gc.enable()

#ENV_NAME = 'InvertedPendulum-v1'
#ENV_NAME = 'Reacher-v1'
ENV_NAME = 'PA-v1'
EPISODES = 100000
TEST = 10
order_list=[[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]];
def main():
    env = filter_env_canonical.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    #env=wrappers.Monitor(env,'experiments2/'+ENV_NAME,force=True);

    for episode in xrange(EPISODES):
        #program_order_idx=np.random.randint(1,4);
        program_order_idx=1;
        env.set_order(program_order_idx,order_list[program_order_idx]);
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
			ts_reward = 0
			for i in xrange(TEST):
                                program_order_idx=0;
                                env.set_order(program_order_idx,order_list[program_order_idx]);
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					ts_reward += reward
					if done:
						break
			ave_ts_reward = ts_reward/TEST/200
			tr_reward = 0
			for i in xrange(TEST):
                                #program_order_idx=np.random.randint(1,4);
                                program_order_idx=1;
                                env.set_order(program_order_idx,order_list[program_order_idx]);
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					tr_reward += reward
					if done:
						break
			ave_tr_reward = tr_reward/TEST/200
                        print 'episode: ',episode,'Unseen Case Average Reward:',ave_ts_reward,'Training Case Average Reward:',ave_tr_reward
                        f=open("logs","a");
                        f.writelines('episode: '+str(episode)+'Unseen Case Average Reward:'+str(ave_ts_reward)+'Training Case Average Reward:'+str(ave_tr_reward)+"\n");
                        f.close();
                        #if(ave_ts_reward>=-3.75):
                        #  print("Done!!");
                        #  break;
    #env.monitor.close()

if __name__ == '__main__':
    main()
