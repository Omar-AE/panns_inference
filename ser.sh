#!/bin/bash

cd /home/nvr/panns_inference
mkdir /home/nvr/airesults
touch /home/nvr/airesults/ser.json
:> /home/nvr/airesults/ser.json
mkdir /home/nvr/converted_audio_files/
source /opt/intel/openvino/bin/setupvars.sh

#taskset --cpu-list 1
/usr/bin/python3 /home/nvr/panns_inference/convert.py &
P1=$!
/usr/bin/python3 /home/nvr/panns_inference/predict.py &
P2=$!
wait $P1 $P2
