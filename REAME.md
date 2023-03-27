# OpenVINO model tools

    WorkPath: [PATH]/ovmtools_accuracychecker/
    .
    ├── openvino
    │   ├── ...
    ├── ovmtools_accuracychecker
    │   ├── test_acc.py
    │   ├── ...

#### Dependencies
```bash
python -m venv python_env && source python_env/bin/activat
sudo apt-get install nfs-common
```
# Accuracy test

    <!-- Mount IR cache by nfs -->
    $ cd $WorkPath/../ && mkdir nfs_share
    $ sudo mount -t nfs 10.67.108.173:/home/vsi/nfs_share ./nfs_share

    <!-- Mount test data -->

    $ python ./test_acc.py

# Performance regression test for CPU plugin

#### run tests

run the script inside `screen` and press `ctrl+A D` to detach it to prevent early termination due to accidental ssh connection lost.

change some variables inside benchmark_test.sh before run it:

 - setup `binA` `binB` to OpenVINO bin folder containing the binary to be validated;
 - setup `nfs_models_cache` to the path of nfs folder mounted earlier
 - setup `ncpus`,`cpus` and `node` according to the server's lscpu result.

```bash
./benchmark_test.sh | tee log.txt
```

use `screen -list` to check detached session and `screen -r xxxx` to re-attach.

#### visualize results

```bash
./benchmark_vis.sh log.txt > vis.txt
```

#### detailed comparison between two builds of openvino

```bash
./compare_pc.sh xxxx.xml
```
this generates `pcA.txt`, `pcB.txt`, `exec_graph_A.xml` and `exec_graph_B.xml`; you can analyze the result with following command:

```bash
./compare_vis.py pcA.txt pcB.txt
```

