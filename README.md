Programmable Agents
====================

Tensorflow Implementation of Programmable Agents from Google Deepmind.

Implementation is on Tensorflow r1.3.

DDPG script is based on songrotek's repo. https://github.com/songrotek/DDPG.git

https://arxiv.org/abs/1706.06383

" The agents learn to ground the terms in this language in their environment,
and can generalize their behavior at test time to execute new programs that refer to
objects that were not referenced during training. The agents develop disentangled
interpretable representations that allow them to generalize to a wide variety of
zero-shot semantic tasks." from the paper

![alt tag](https://github.com/jaesik817/programmable-agents_tensorflow/blob/master/figures/pa.PNG)

Environment Settings
-----------------

This module uses OpenAI GYM. for installing that, please refer https://github.com/openai/gym.

This module uses Mujoco physics engine and mujoco_py 0.5 branch from OpenAI. For installing that, please refer https://github.com/openai/mujoco-py.
(`pip install mujoco-py==0.5.7`)

For adding this module on your gym, you need to run bellowed bash script.

`
bash run_require.sh
`

Then you can check followed lines are added in <python path>/dist-packeges/gym/envs/__init__.py and <python path>/dist-packages/gym/envs/mujoco/__init__.py respectively.


    register(
    
    id='PA-v1',

    entry_point='gym.envs.mujoco:PAEnv',

    max_episode_steps=50,

    reward_threshold=-3.75,
    
    )


`
from gym.envs.mujoco.pa import PAEnv
`

You also can check two files(pa.py and pa.xml) are copyed in gym folder.

Note that run_require.sh script adds above lines on your gym configure, thus please do not running that more than one time. 

 Multiple Object Reacher
-----------------

For running programmable nets,

`
python gym_ddpg.py
`

For running canonical nets,

`
python gym_ddpg_canonical.py
`

### Settings
In the environment, there are 4 objects, which are red/blue cube and sphere, respectively.

Arm has two joints for making experiment more simple than the paper's one.

Blue cube, red sphere and blue sphere are used for training, and red cube is used for zero-shot testing.

The feature (row of capital theta) is 6 (Red, Blue, White, Cube, Sphere, Hand).

The state dimension is 75 (5(4 objects and arm) * 15 (3/6/6 dimension are for position, feature, and feature of arm.)).

