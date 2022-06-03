#!/usr/bin/python3
import numpy as np
#import matplotlib.pyplot as plt
import sys, os

log_file = sys.argv[1]

stat = {}

with open(log_file,"r") as f:
    for l in f.readlines():
        logs = l.split(' ')

        if len(logs) != 3:
            print("[ERROR] {}".format(l.rstrip("\n").rstrip("\r")))
            continue
        fpsA, fpsB, xml = l.split(' ')

        ratio = (float(fpsB)/float(fpsA)) - 1.0
        fullpath = xml.rstrip("\n").rstrip("\r")
        if not fullpath in stat:
            stat[fullpath] = [[],[]]
        
        stat[fullpath][0].append(ratio)
        stat[fullpath][1].append([fpsA, fpsB])

summary = sorted(stat.items(), key=lambda d: np.mean(d[1][0]), reverse=True)

cnt = 0
for k, v in summary:
    ratios = np.array(v[0])
    r_mean = np.mean(ratios)
    r_min = np.amin(ratios)
    r_max = np.amax(ratios)
    name = os.path.split(k)[1]

    print("[{}] {:>8.1f}% {}% {}".format(cnt, r_mean*100, (ratios*100).astype(np.int32), k))
    for fps in v[1]:
        print("        {} : {}".format(fps[0], fps[1]))
    cnt+=1

print("Total {} modles".format(cnt))