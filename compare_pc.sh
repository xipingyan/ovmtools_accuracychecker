#!/bin/bash

binA=`pwd`/../openvino/bin/intel64/Release
binB=~/openvino/bin-new2.6B/intel64/Release
binB=$binA

args=" -t 5 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32"
nfs_models_cache=~/models/nfs/models_cache

cores=4,5,6,7
node=0

#models=`find ${nfs_models_cache} -name *.xml`

LD_LIBRARY_PATH=${binB}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binB}/benchmark_app -m $1 $args -exec_graph_path exec_graph_B.xml |& tee pcB.txt
LD_LIBRARY_PATH=${binA}/lib ONEDNN_VERBOSE=0 taskset -c $cores numactl -C $cores -m $node -- /usr/bin/time -v ${binA}/benchmark_app -m $1 $args -exec_graph_path exec_graph_A.xml |& tee pcA.txt

echo A = ${binA}
echo B = ${binB}

python3 compare_vis.py pcA.txt pcB.txt