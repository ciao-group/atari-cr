# Computational Rationality for Atari Games
This repository aims to provides agents that play Atari games with human-plausible active vision behavior. 

## Prerequisites
- Linux or MacOS system, Windows is NOT supported
- python (recommended version: 3.11)
## Installation
All four installation steps are required
### System dependencies
- The following system wide dependencies are needed. Exact names on non-debian distros may vary.
- cmake version 4 is not supported. Installl cmake version 3
``` sh
sudo apt install cmake libx11-dev libglew-dev patchelf build-essential zlib1g-dev libglib2.0-0
```
### Project Installation
``` sh
git clone https://github.com/ciao-group/atari-cr
cd atari-cr
pip install -e .
```
### ROMs
Install Atari ROMs by running
```bash
curl -L https://www.atarimania.com/roms/Atari-2600-VCS-ROM-Collection.zip -o x.zip && unzip x.zip
python -m atari_py.import_roms ROMS && rm -r ROMS 'HC ROMS' x.zip
```
or follow the steps described in https://github.com/openai/atari-py
<!-- ## Mujoco
- Follow the steps described in https://github.com/openai/mujoco-py -->
### Dataset
Download and decompress the Atari-HEAD dataset:
```sh
./data/Atari-HEAD/download.sh
```

## Entry points
- Run `python src/atari_cr/agents/dqn_atari_cr/main.py` to train an agent. See `python src/atari_cr/agents/dqn_atari_cr/main.py --help` for viable arguments.
- Use `python src/atari_cr/hyperparams.py` to test multiple agents with different hyperparameters.
- Use `./run.sh` to start hyperparameter tuning together with tensorboard.

## Acknowledgement
We thank the authors of [SugaRL](https://github.com/elicassion/sugarl) from which this repository was forked.
