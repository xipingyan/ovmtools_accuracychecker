
binA=~/openvino/bin-master/intel64/Release
binB=~/openvino/bin-new2.6/intel64/Release
nfs_models_cache=~/models/nfs/models_cache

ncpus=4
cpus=60,61,62,63
node=1

args="-t 10 -nireq=2 -nstreams 1 -nthreads ${ncpus}  -infer_precision f32"
models=`find ${nfs_models_cache} -name "*.xml"`

for i in {0..50}; do
for m in $models; do
for t in {0..2}; do
ThroughputA=`LD_LIBRARY_PATH=${binA}/lib numactl -C $cpus -m $node -- ${binA}/benchmark_app -m $m $args | grep Throughput`
ThroughputB=`LD_LIBRARY_PATH=${binB}/lib numactl -C $cpus -m $node -- ${binB}/benchmark_app -m $m $args | grep Throughput`

fpsA="$(echo $ThroughputA | cut -d' ' -f5)"
fpsB="$(echo $ThroughputB | cut -d' ' -f5)"

echo $fpsA $fpsB $m
done
done
done