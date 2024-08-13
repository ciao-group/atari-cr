#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

envlist=(ms_pacman road_runner breakout)
expname="pauseable130_1m"
logdir=output/${expname}/logs

for i in ${!envlist[@]}
do
    gpuid=$(( $i % $numgpus ))
    (
        for seed in 0 1 2
        do
            echo "${expname} GPU: ${gpuid} Env: ${envlist[$i]} Seed: ${seed} ${1}"
            basename=$(basename $1)
            CUDA_VISIBLE_DEVICES=$gpuid python $1 \
                --clip_reward \
                --capture_video \
                \
                --env ${envlist[$i]} \
                --seed $seed \
                --exp_name ${expname} \
                --total_timesteps 1000000 \
                \
                --use_pause_env \
                --pause_cost 0.10 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 \
                --fov_size 20 \
                --no_action_pause_cost 0.2 \
                &
        done
    ) &
done
wait
