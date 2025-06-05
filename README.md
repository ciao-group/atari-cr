# Computational Rationality for Atari Games
This repository aims to provides agents that play Atari games with human-plausible active vision behavior. 

## Prerequisites
- Linux or MacOS system
- python (recommended version: 3.11)
## Installation
All three installation steps are required
### Project Installation
``` sh
git clone https://github.com/ciao-group/atari-cr
cd atari-cr
pip install -e .
```
### ROMs and Mujoco
- Install Atari ROMs by following the steps described in https://github.com/openai/atari-py
- Follow the steps described in https://github.com/openai/mujoco-py
### System dependencies
- The following system wide dependencies are needed. Exact names on non-debian distros may vary.
``` sh
sudo apt install libx11-dev libglew-dev patchelf python-dev
```

## Acknowledgement
We thank the authors of [SugaRL](https://github.com/elicassion/sugarl) from which this repository was forked.
