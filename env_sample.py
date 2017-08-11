#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import math
import time
import os
import six
import numpy as np
import mujoco_py
from mujoco_py import MjSim, MjViewer

epi_num=1000;
time_limit=10e5;
# Model Loading
model = mujoco_py.load_model_from_path("xmls/env.xml");
sim = MjSim(model);
# Programmable Agent
agent = pa();
for i in range(epi_num):
  # Sim variable reset
  sim.reset();
  for j in range(time_limit):
    fea_mat=agent.get_fea_mat(sim);
    action = agent.action(fea_mat);
    sim.data.ctrl[:]=action;sim.step();
    reward,done = agent.get_reward(sim);
    if done:
      break;

  
sim.data.ctrl[:]=0.1;
#print(sim.data.get_body_xpos("fingertip"));exit(1);
viewer = MjViewer(sim)
step = 0
while True:
    sim.step()
    viewer.render()
    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
