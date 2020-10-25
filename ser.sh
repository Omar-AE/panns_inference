#!/bin/bash

ctrl_c() {
        echo "Done";
        kill -9 $P1
	kill -9 $P2
        exit;
}

trap ctrl_c SIGINT


cd /home/nvr/panns_inference
mkdir /home/nvr/airesults
touch /home/nvr/airesults/ser.json
:> /home/nvr/airesults/ser.json
mkdir /home/nvr/converted_audio_files/
source /opt/intel/openvino/bin/setupvars.sh


taskset --cpu-list 1 /usr/bin/python3 /home/nvr/panns_inference/convert.py &
P1=$!
taskset --cpu-list 1 /usr/bin/python3 /home/nvr/panns_inference/predict.py &
P2=$!

while true  # to keep the bash file running so that pressing ctrl+c kills all running python scripts
do
	sleep 100
done
