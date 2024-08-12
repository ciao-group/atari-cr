#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

envlist=(boxing)
expname="pauseable129_1m"
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
                --pause_cost 0.05 \
                --fov_size 20 \
                &
        done
    ) &
done
wait
