#!/bin/bash

# envs init file modification
cp /usr/local/lib/python3.5/dist-packages/gym/envs/__init__.py requirement_files/env_init.py
cat requirement_files/env_init >> requirement_files/env_init.py
sudo mv requirement_files/env_init.py /usr/local/lib/python3.5/dist-packages/gym/envs/__init__.py

# mujoco init file modification
cp /usr/local/lib/python3.5/dist-packages/gym/envs/mujoco/__init__.py requirement_files/mojoco_init.py
cat requirement_files/mojoco_init >> requirement_files/mujoco_init.py
sudo mv requirement_files/mujoco_init.py /usr/local/lib/python3.5/dist-packages/gym/envs/mujoco/__init__.py

# copy pa agent python script
cp requirement_files/pa.py /usr/local/lib/python3.5/dist-packages/gym/envs/mujoco/

# copy pa xml file
cp requirement_files/pa.xml /usr/local/lib/python3.5/dist-packages/gym/envs/mujoco/assets/
