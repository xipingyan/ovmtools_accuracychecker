#!/usr/bin/python3

import numpy as np
#import matplotlib.pyplot as plt
import sys, os
import argparse
from xml.dom.minidom import parse
import xml.dom.minidom

exec_graphA = ''
exec_graphB = ''
args = None

def find_layout(exec_graph, name):
    if (name.endswith("...")):
        tag = 'originalLayersNames="{}'.format(name.rstrip("..."))
    else:
        tag = 'originalLayersNames="{}'.format(name)
    layout='outputLayouts="'
    found = 0
    ret = "?"
    for l in exec_graph:
        if tag in l:
            sl = l[l.index(layout) + len(layout):]
            found += 1
            ret = sl[0:sl.index('"')]
    
    if found > 1:
        ret = "?"
    return ret

if args and len(args.node_type) > 0 and args.node_cnt <= 0:
    args.node_cnt = 99999999

pc_log_start_tag = "[ INFO ] Performance counts for 0-th infer request:"
pc_log_end_tag="Total time:"


class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

def analyse(log_file):
    pc_by_type = {}
    pc_by_node = {}
    
    stat = []
    with open(log_file,"r") as f:
        start  = False
        for l in f.readlines():
            if l.startswith("	Percent of CPU this job got") or \
               l.startswith("	Maximum resident set size (kbytes)") or \
               l.startswith("	User time (seconds)"):
                stat.append(l.strip("\t"))
                continue
            if l.startswith("[ INFO ] 	Average:") or \
               l.startswith("[ INFO ] Throughput:"):
                stat.append(l[9:].strip(" ").strip("\t"))
                continue

            if l == '\n': continue
            if l.startswith(pc_log_start_tag):
                start = True
                continue
            if start:
                if l.startswith(pc_log_end_tag):
                    start = False
            if start:
                name = l[:30].rstrip(" ")
                run, _, layer_type, _, realTime, _, cpuTime, _, execType = l[30:].split()

                if (run == "NOT_RUN"):
                    continue
                node_type = layer_type + "_" + execType

                if node_type.startswith('Convolution_brgconv_avx512'):
                    node_type = node_type.replace('brgconv', 'brg.jit')
                elif node_type.startswith('Convolution_jit_avx512'):
                    node_type = node_type.replace('jit', 'brg.jit')
                if node_type.startswith('GroupConvolution_ref_any_FP32'):
                    node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
                elif node_type.startswith('GroupConvolution_brgconv_avx512') and node_type.endswith('_FP32'):
                    node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
                elif node_type.startswith('GroupConvolution_jit_avx512_FP32'):
                    node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'
                elif node_type.startswith('GroupConvolution_jit_gemm_FP32'):
                    node_type = 'GroupConvolution_any.brg.jit.gemm_FP32'

                if not node_type in pc_by_type:
                    pc_by_type[node_type] = [0,0] # cnt, total

                pc_by_type[node_type][0] += 1
                pc_by_type[node_type][1] += int(realTime)

                pc_by_node[name] = [int(realTime), layer_type, execType]

    pc_by_node = sorted(pc_by_node.items(), key=lambda d: d[1][0], reverse=True)
    pc_by_type = sorted(pc_by_type.items(), key=lambda d: d[1][1], reverse=True)
    return pc_by_node, pc_by_type, stat


def show_result(log_file, pc_by_node, pc_by_type, stat):
    if args.node_cnt > 0:
        print("=========== {} by node =============".format(log_file)) 
        for i, (name, (total, *other)) in enumerate(pc_by_node):
            if i >= args.node_cnt:
                break
            if total == 0:
                break
            print("{:>8} us {} {}".format(total, name, other))

    total_time = 0
    print("=========== {} by type =============".format(log_file)) 
    for type_name, (cnt, total) in pc_by_type:
        total_time += total
        if total == 0:
            break
        print("{:>8.1f} x {:<5} us {}".format(total/cnt, cnt, type_name))

    print("total_time:{}".format(total_time))
    for s in stat:
        print(s.rstrip("\n").rstrip("\r"))

def smart_val(v):
    if abs(v) > 1000000:
        return "{:.1f}M".format(v/1000000)
    if abs(v) > 1000:
        return "{:.1f}K".format(v/1000)
    return v


def choose_color(t0, t1):
    color_start = Colors.DARK_GRAY
    if t1 > t0 * 1.05:
        color_start = Colors.LIGHT_RED
    if t1 < t0 * 0.95:
        color_start = Colors.GREEN
    color_end = Colors.END
    return color_start, color_end



def show_compare_result(log_fileA, log_fileB):

    pc_by_node0, pc_by_type0, stat0 = analyse(log_fileA)
    pc_by_node1, pc_by_type1, stat1 = analyse(log_fileB)
    

    print("{}   :    {}".format(log_fileA, log_fileB))
    print("*********************************************************")
    print("*                   comparing by type                   *")
    print("*********************************************************")
    # collect all type names
    type_names = [t for t, _ in pc_by_type0]
    for t, _ in pc_by_type1:
        if not t in type_names:
            type_names.append(t)

    def find(pclist, type_name):
        for name, v in pclist:
            if name == type_name:
                return v
        return None

    total_time0 = total_time1 = 0
    results = []
    for type_name in type_names:
        v0 = find(pc_by_type0, type_name)
        if v0:
            cnt, time0 = v0
            total_time0 += time0
            info0 = "{:.1f} x {}".format(time0/cnt, cnt)
        else:
            time0 = 0
            info0 = "---"

        v1 = find(pc_by_type1, type_name)
        if v1:
            cnt, time1 = v1
            total_time1 += time1
            info1 = "{:.1f} x {}".format(time1/cnt, cnt)
        else:
            time1 = 0
            info1 = "---"
        
        color_start, color_end = choose_color(time0, time1)
        print("{} {:>8} {:>32}   {:<32}   {} {}".format(color_start, smart_val(time1-time0),  info0, info1, type_name, color_end))
        results.append((type_name, (time1-time0)/1000))

    color_start, color_end = choose_color(total_time0, total_time1)
    print("")
    print("{:>8}  {:>32}   {:<32}   {}".format(smart_val(total_time1 - total_time0), total_time0, total_time1, "Totals"))

    node_cnt_to_show = args.node_cnt if args else 0
    if (node_cnt_to_show > 0):
        print("*********************************************************")
        print("*                   comparing by node                   *")
        print("*********************************************************")
        # collect all type names
        all_names = [t for t, _ in pc_by_node0]
        for t, _ in pc_by_node1:
            if not t in all_names:
                all_names.append(t)

        # 
        total_time0 = total_time1 = 0
        for name in all_names:
            if (node_cnt_to_show <= 0):
                break
            v0 = find(pc_by_node0, name)
            if v0:
                time0, layer0, exectype = v0
                
                info0 = "{}_{}_{} {:6.1f}".format(layer0, exectype, find_layout(exec_graphA, name), time0)
            else:
                time0 = 0
                info0 = "---"

            v1 = find(pc_by_node1, name)
            if v1:
                time1, layer1, exectype = v1
                info1 = "{:<6.1f} {}_{}_{}".format(time1, layer1, exectype, find_layout(exec_graphB, name))
            else:
                time1 = 0
                info1 = "---"
            
            if args.node_type in info0 or args.node_type in info1:
                node_cnt_to_show -= 1
                total_time0 += time0
                total_time1 += time1
                color_start, color_end = choose_color(time0, time1)
                print("{} {:>6} {:>50}  {:<50}  {} {}".format(color_start, smart_val(time1-time0), info0, info1, name, color_end))

        color_start, color_end = choose_color(total_time0, total_time1)
        print("")
        print("{}{:>6} {:>50}   {:<50}   {}{}".format(color_start, smart_val(total_time1 - total_time0), total_time0, total_time1, "Totals", color_end))

    print("")
    for i in range(len(stat0)):
        s0 = stat0[i].rstrip("\n").rstrip("\r")
        s1 = stat1[i].rstrip("\n").rstrip("\r")
        print("{:>50}   {:<50} ".format(s0, s1))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--node_cnt", help="number of nodes to show", default=10, type=int)
    parser.add_argument("-t","--node_type",help="node type filter",default="", type=str)
    parser.add_argument("log_fileA", nargs="?")
    parser.add_argument("log_fileB", nargs="?")
    parser.add_argument("exec_graphA", nargs="?")
    parser.add_argument("exec_graphB", nargs="?")

    args = parser.parse_args()
    with open(args.exec_graphA or 'exec_graph_A.xml') as f:
        exec_graphA = f.readlines()

    with open(args.exec_graphB or 'exec_graph_B.xml') as f:
        exec_graphB = f.readlines()

    show_compare_result(args.log_fileA or 'pcA.txt', args.log_fileB or 'pcB.txt')

#show_result(args.log_files[0], pc_by_node0, pc_by_type0, stat0)
