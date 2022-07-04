cpus=4,5,6,7
node=0
new_log=brg
base_log=jit
model_dir=/home/luocheng/models_cache/models_cache/
bin_dir=`pwd`/../openvino/bin/intel64/Release

export ONEDNN_VERBOSE=0
# without brg
numactl -C $cpus -m $node -- python3 test_all.py $bin_dir $base_log $model_dir
# with brg
numactl -C $cpus -m $node -- python3 test_all.py $bin_dir $new_log $model_dir -expconv
numactl -C $cpus -m $node -- python3 postprocess.py $new_log.log $base_log.log -0.1 $model_dir $bin_dir
