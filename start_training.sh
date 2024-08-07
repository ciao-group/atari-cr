#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

envlist=(boxing freeway)
expname="pauseable125_1m"
logdir=output/${expname}/logs

for i in ${!envlist[@]}
do
    gpuid=$(( $i % $numgpus ))
    (
        for seed in 1
        do
            echo "${expname} GPU: ${gpuid} Env: ${envlist[$i]} Seed: ${seed} ${1}"
            basename=$(basename $1)
            CUDA_VISIBLE_DEVICES=$gpuid python $1 \
                --env ${envlist[$i]} \
                --seed $seed \
                --exp-name ${expname} \
                --fov-size 20 \
                --clip-reward \
                --capture-video \
                --total-timesteps 1000000 \
                --buffer-size 100000 \
                --learning-start 80000 \
                \
                --sensory-action-mode absolute \
                --grokfast \
                \
                --no-action-pause-cost 0.1 \
                --no-pause-env \
                --pvm-stack 6 \
                --pause-cost 0.01 \
                --successive-pause-limit 30
        done
    ) &
done
wait
