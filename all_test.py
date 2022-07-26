import subprocess
import os
import sys
from pathlib import Path

cwd = os.getcwd()
binA = cwd + '/../openvino/bin/intel64/RelWithDebInfo'
binB = cwd + '/../openvino-tmp/bin/intel64/RelWithDebInfo'
args = []
args = [str(i) for i in args]
base = 'c:/models/'
if len(sys.argv) > 1:
    binA = sys.argv[1]
output_base = 'test_all'
if len(sys.argv) > 2:
    output_base = sys.argv[2]
if len(sys.argv) > 3:
    base = sys.argv[3]
if len(sys.argv) > 4:
    args += sys.argv[4:]

os.chdir(binA)
result = []
log = open(f'{cwd}/{output_base}_detail.log', 'w')
out_f = open(f'{cwd}/{output_base}.log', 'w')
pathlist = Path(base).rglob('*.xml')
for idx, path in enumerate(pathlist):
    f = str(path)
    print(f'{idx:3d} {f}... ', end=''),
    for i in range(3):
        outputA = subprocess.run(['./benchmark_app', '-m', f] + args, capture_output=True)
        out = outputA.stdout.decode()
        log.write(out)
        if outputA.returncode == 0:
            fps = out.split('Throughput')[-1]
            fps = fps.split()[1]
            line = f'{fps} {f}'
            result.append(line)
            print(f'{fps}', end=' '),
        else:
            line = f'-1 {f}'
            result.append(line)
        out_f.write(line + '\n')
        out_f.flush()

    print(' ')

# with open(f'{cwd}/{output_base}.log', 'w') as f:
#     f.write('\n'.join(result))