#! /bin/bash

set -e
ROOT="$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH=${ROOT}:${PYTHONPATH}
victim_model="mf"
attack_model="average aush aia aushplus"

cd $ROOT
pip install -e .

for am in $attack_model; do
    for vm in $victim_model; do
        echo "Running $vm, $am in dev"
        recad_runner --tqdm=0 --filler_num=0 --data=dev --victim=$vm --attack=$am --rec_epoch=1 --attack_epoch=1 
    done
done
