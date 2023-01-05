import sys

physcpubind="0,2,4,6,8"
ov_bin_folder="/home/tingqian/openvino/bin/intel64/Release"
model_base="/home/openvino-local-compile-02/common_data/sk_13sept_75models_22.2_int8/"

accuracy_checker_configs = "/home/openvino-local-compile-02/common_data/accuracy-checker/configs/omz_validation/"
definitions_file=f"{accuracy_checker_configs}/dataset_definitions.yml"

data_source=f"/home/tingqian/odt_nfs_share/omz-validation-datasets/"
model_attributes=f"{data_source}/model_attributes"

annotations=f"/home/tingqian/task/remove_binary_postops/annotations"

# 
# sudo mount -t nfs 10.67.107.130:/home/share/  ./odt_nfs_share/
# 
#
#
if __name__ == "__main__":
    all_variables = dir()
    print(eval(sys.argv[1]))
