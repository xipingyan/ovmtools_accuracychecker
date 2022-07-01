import os
import sys
import subprocess
import compare_vis as compare_vis

# new_log base_log low_thresh model_base_dir ov_bench_dir
new_log = sys.argv[1]
base_log = sys.argv[2]
thresh = float(sys.argv[3])
model_base_dir = sys.argv[4]
ov_bench_dir = sys.argv[5]

args = ['-t', 5, '-b', 1, '-nireq=1', '-nstreams=1', '-nthreads=4', '-infer_precision=f32', '-pc']
args = [str(i) for i in args]

def median(path):
    with open(path, 'r') as f:
        c = f.readlines()
    results = []
    results_data = []
    for i in range(len(c) // 3):
        v0 = float(c[0 + 3 * i].split()[0])
        v1 = float(c[1 + 3 * i].split()[0])
        v2 = float(c[2 + 3 * i].split()[0])
        #v = v0 + v1 + v2 - max([v0, v1, v2]) - min([v0, v1, v2])
        v = max([v0, v1, v2])
        results.append(f'{v:.2f},{c[0 + 3 * i].split()[1]}')
        results_data.append((v, c[0 + 3 * i].split()[1]))
    with open(f'{path}.max.csv', 'w') as f:
        f.write('\n'.join(results))
    return results_data
# 1, filter the performance data
result_new = median(new_log)
result_base = median(base_log)

# 2, filter the desired sets
assert(len(result_new) == len(result_base))
result_sets = []
for i in range(len(result_new)):
    if (result_new[i][0] - result_base[i][0]) / result_base[i][0] < thresh:
        # idx, name, new_fps, base_fps
        result_sets.append((i, result_new[i][1], result_new[i][0], result_base[i][0]))

# 3, run compare tool
detail_f = open(f'{new_log}.layer.csv', 'w')
detail_f.write('name,new_fps,base_fps,ratio,layer1,time1(ms),,,,,,,,,,\n')
for (i, name, new_fps, base_fps) in result_sets:
    os.environ['USE_BRG'] = '0'
    outputA = subprocess.run([f'{ov_bench_dir}/benchmark_app', '-m', model_base_dir + name] + args, capture_output=True)
    out = outputA.stdout.decode()
    if outputA.returncode == 0:    
        with open('test1.log', 'w') as f:
            f.write(out)
    os.environ['USE_BRG'] = '1'
    outputA = subprocess.run([f'{ov_bench_dir}/benchmark_app', '-expconv', '-m', model_base_dir + name] + args, capture_output=True)
    out = outputA.stdout.decode()
    if outputA.returncode == 0:
        with open('test2.log', 'w') as f:
            f.write(out)
    result = compare_vis.show_compare_result('test2.log', 'test1.log')
    detail_f.write(f'{name},{new_fps},{base_fps},{(new_fps-base_fps)/base_fps:.3f},')
    result = sorted(result, key=lambda x: x[1])
    for (n, t) in result:
        if t < 0:
            detail_f.write(f'{n},{t},')
    detail_f.write('\n')