#!/bin/bash
numgpus=${2:-$(nvidia-smi --list-gpus | wc -l)}

envlist=(boxing freeway)

expname="pauseable125_1m"
totaltimesteps="1000000"
buffersize="100000"
learningstarts="10000"

logdir=output/${expname}/logs

for i in ${!envlist[@]}
do
    gpuid=$(( $i % $numgpus ))
    (
        for seed in 0
        do
            echo "${expname} GPU: ${gpuid} Env: ${envlist[$i]} Seed: ${seed} ${1}"
            basename=$(basename $1)
            CUDA_VISIBLE_DEVICES=$gpuid python $1 \
                --env ${envlist[$i]} \
                --env-num 1 \
                --seed $seed \
                --exp-name ${expname} \
                --fov-size 20 \
                --clip-reward \
                --capture-video \
                --total-timesteps $totaltimesteps \
                --buffer-size $buffersize \
                --learning-start $learningstarts \
                --pvm-stack 6 \
                --pause-cost 0.01 \
                --successive-pause-limit 30 \
                --no-action-pause-cost 0.1 \
                --sensory-action-mode relative \
                --grokfast \
                --no-pause-env
        done
    ) &
done
wait
