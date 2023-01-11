cpus=4,5,6,7
node=0
new_log=brg.2.8
base_log=jit.2
#model_dir=/home/luocheng/models_cache/models_cache/
model_dir=/workpath/models/instance-segmentation-security-0228/fp32/
bin_dir=`pwd`/../openvino/bin/intel64/Release

export ONEDNN_VERBOSE=0
# without brg
python3 test_all.py $bin_dir $base_log $model_dir -dopt
# with brg

#numactl -C $cpus -m $node -- python3 test_all.py $bin_dir $new_log $model_dir -cpu_experimental=brgconv -dopt
#numactl -C $cpus -m $node -- python3 postprocess.py $new_log.log $base_log.log -0.05 $model_dir $bin_dir
#numactl -C $cpus -m $node -- python3 postprocess.py $base_log.log $new_log.log -0.05 $model_dir $bin_dir check_fast
