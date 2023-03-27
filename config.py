import sys

physcpubind="0,2,4,6,8"

#ov_bin_folder="/home/tingqian/openvino/bin/intel64/Release"
#model_base="/home/openvino-local-compile-02/common_data/sk_13sept_75models_22.2_int8/"
#accuracy_checker_configs = "/home/openvino-local-compile-02/common_data/accuracy-checker/configs/omz_validation/"
#definitions_file=f"{accuracy_checker_configs}/dataset_definitions.yml"
#data_source=f"/home/tingqian/odt_nfs_share/omz-validation-datasets/"
#model_attributes=f"{data_source}/model_attributes"
#annotations=f"/home/tingqian/task/remove_binary_postops/annotations"

# Mount test_data and models folder.
# ================================================
# sudo mkdir -p ../nfs_share
# sudo mkdir -p ../nfs_share_2
# Mount model path:
# sudo mount -t nfs 10.67.108.173:/home/vsi/nfs_share ../nfs_share
# Mount test data path:
# sudo mount -t nfs 10.67.107.130:/home/share ../nfs_share_2

# Only accept full path.
WORK_PATH="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/ovmtools_accuracychecker"
ov_bin_folder=f"{WORK_PATH}/../openvino/bin/intel64/Release"
model_base=f"{WORK_PATH}/../nfs_share/sk_13sept_75models_22.2_int8/"
model_base=f"{WORK_PATH}/../nfs_share/ww09_weekly_23.0.0-9828-4fd38844a28-API2.0-FP16/"
model_base=f"{WORK_PATH}/../nfs_share/ww10_weekly_23.0.0-9926-63d282fd73c-API2.0/"

accuracy_checker_configs = f"{WORK_PATH}/../openvino/thirdparty/open_model_zoo/tools/accuracy_checker/"
definitions_file=f"{accuracy_checker_configs}/dataset_definitions.yml"

data_source=f"{WORK_PATH}/../nfs_share_2/omz-validation-datasets/"
model_attributes=f"{data_source}/model_attributes"

annotations=f"{WORK_PATH}/../dataset/data_cache/annotations"

if __name__ == "__main__":
    all_variables = dir()
    print(eval(sys.argv[1]))

