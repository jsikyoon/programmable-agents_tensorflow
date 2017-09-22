import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class PAEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pa.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("hand")-self.get_body_com("target1")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        for i in range(1,5):
          while True:
              self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
              if np.linalg.norm(self.goal) < 2:
                  break
          qpos[2*i:2*(i+1)] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        self.object_num=5;
        self.fea_size=15;
        Omega=np.zeros((self.fea_size,self.object_num),dtype=float);
        # position
        position=self.get_body_com("target1");
        Omega[:3,0]=self.get_body_com("hand")-position;
        #Omega[:3,0]=position;
        position=self.get_body_com("target2");
        Omega[:3,1]=self.get_body_com("hand")-position;
        #Omega[:3,1]=position;
        position=self.get_body_com("target3");
        Omega[:3,2]=self.get_body_com("hand")-position;
        #Omega[:3,2]=position;
        position=self.get_body_com("target4");
        Omega[:3,3]=self.get_body_com("hand")-position;
        #Omega[:3,3]=position;
        position=self.get_body_com("hand");
        Omega[:3,4]=self.get_body_com("hand")-position;
        #Omega[:3,4]=position;
        # color one-hot encoding 3D (red,blue,white)
        Omega[3,0]=1.0;
        Omega[3,2]=1.0;
        Omega[4,1]=1.0;
        Omega[4,3]=1.0;
        Omega[5,4]=1.0;
        # shape one-hot encoding 3D (cube,sphere,hand)
        Omega[6,0]=1.0;
        Omega[6,1]=1.0;
        Omega[7,2]=1.0;
        Omega[7,3]=1.0;
        Omega[8,4]=1.0;
        # arm state
        theta = self.model.data.qpos.flat[:2];
        for i in range(self.object_num):
          Omega[9:11,i]=np.cos(theta);
          Omega[11:13,i]=np.sin(theta);
          Omega[13:15,i]=self.model.data.qvel.flat[:2];
        """  
        #qpos
        for i in range(self.object_num):
          Omega[3:13,i]=self.sim.data.qpos.flat;
        """
        Omega=np.transpose(Omega);
        Omega=np.reshape(Omega,[(self.object_num)*self.fea_size]);
        return Omega;

    # reacher-v1 state feature
    def _get_obs2(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
