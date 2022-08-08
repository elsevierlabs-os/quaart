#!/bin/bash
if [ -z "$1" ]
then
    set -- "results"
fi

./data-prep-script.sh
./experiment1.sh hands_$1
./experiment2.sh figer_$1
#echo figer_$1
