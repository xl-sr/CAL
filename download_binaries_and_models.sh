#!/usr/bin/env bash

# download and unpack the CARLA 0.8.2 binaries
gdown https://drive.google.com/uc\?id\=1ZtVt1AqdyGxgyTm69nzuwrOYoPUn_Dsm
tar --extract --file=CARLA_0.8.2.tar.gz Engine
tar --extract --file=CARLA_0.8.2.tar.gz CarlaUE4
tar --extract --file=CARLA_0.8.2.tar.gz CarlaUE4.sh
rm -rf CARLA_0.8.2.tar.gz

# download the model weights
gdown https://drive.google.com/uc\?id=1mIi83YjB-0Mfole8W6lSuJRrcy1zauSV
unzip data.zip
mv data PythonClient/agents/CAL_agent/perception
