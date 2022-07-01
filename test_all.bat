set ONEDNN_VERBOSE=0
rem python test_all.py %cd%\../openvino-tmp/bin/intel64/RelWithDebInfo test_base
set USE_BRG=1
python test_all.py %cd%\../openvino/bin/intel64/RelWithDebInfo test_2.6.brg
set USE_BRG=0
python test_all.py %cd%\../openvino/bin/intel64/RelWithDebInfo test_2.6.jit
