#!/bin/bash

# First set of loops
for i in channel; do
  for j in fno unet; do
    for z in 2 4 6 8; do
      echo "Running: python main_bench.py --dataset $i --model_type $j --scale $z"
      python main_bench.py --dataset $i --model_type $j --scale $z
    done
  done
done

# Second set of loops (runs after the first set completes)
for i in channel; do
  for z in 2 4 6 8; do
    echo "Running: python main.py --dataset $i --scale $z"
    python main.py --dataset $i --scale $z
  done
done