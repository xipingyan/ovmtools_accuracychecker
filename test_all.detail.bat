rem step1: test all models
set ONEDNN_VERBOSE=0
rem python test_all.py %cd%\../openvino-tmp/bin/intel64/RelWithDebInfo test_base
set USE_BRG=1
set new_log=test_2.6.loop-oc-L2-2
set base_log=test_2.6.jit.2
python test_all.py %cd%\../openvino/bin/intel64/RelWithDebInfo %new_log% c:/models/
rem python test_all_brg.py %cd%\../openvino/bin/intel64/RelWithDebInfo test_2.6.loop2
set USE_BRG=0
python test_all.py %cd%\../openvino/bin/intel64/RelWithDebInfo %base_log% c:/models/

rem step2: filter test results(select best performance)
python postprocess.py %new_log%.log %base_log%.log -0.1 c:/models/  %cd%/../openvino/bin/intel64/RelWithDebInfo/
rem step3: select the models which drops 10%+

rem step4: use compare tool to find the top drop layers, format: name, base perf, new perf, drop, top1 layer type, top1 delta, ...

rem step5: use compare