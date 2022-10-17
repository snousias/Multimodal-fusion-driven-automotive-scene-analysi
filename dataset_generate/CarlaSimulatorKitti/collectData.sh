#!/usr/bin/env bash

if [ -d "dataset/" ]
then
    echo "Directory dataset exists."
    rm -r dataset/
else
    echo "Directory dataset does not exists."
fi

python generator.py --vehicle-num 200 --time-limit 10 --walker-num 100 --town Town10HD_Opt --image-type png
