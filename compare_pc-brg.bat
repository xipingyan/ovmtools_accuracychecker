@set binA=%cd%\../openvino/bin/intel64/RelWithDebInfo
@set binB=%cd%\../openvino-tmp/bin/intel64/RelWithDebInfo
@set binB=%binA%
set USE_BRG=0

@set args=-niter 100 -b 1 -nireq=1 -nstreams 1 -nthreads 4 -pc -infer_precision f32

@set cur=%cd%
@set ONEDNN_VERBOSE=2
@cd %binB%
benchmark_app -m %1 %args% -exec_graph_path %cur%\exec_graph_B.xml > %cur%\pcB.txt
@cd %binA%
set USE_BRG=1
benchmark_app -m %1 %args% -exec_graph_path %cur%\exec_graph_A.xml > %cur%\pcA.txt
@cd %cur%

@python compare_vis.py pcA.txt pcB.txt