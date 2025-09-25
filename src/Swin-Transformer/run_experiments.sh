#! /bin/bash

imagenet_path=/path/to/imagenet

inner_loop() {
    local bits=$1
    local topk_percent=$2
    local threshold=$3
    local profiling=$4
    local oracle=$5
    local baseline=$6
    local cache=$7
    local MODEL=$8

    uuid=$(uuidgen)
    log_dir=logs/$MODEL/$uuid
    mkdir -p $log_dir
    export bits=$bits
    export topk_percent=$topk_percent
    export threshold=$threshold
    export kqa_threshold=1
    export kqa_topk_prop=1
    export profiling=$profiling
    export oracle=$oracle
    export cache_channel_selection=$cache
    export baseline=$baseline
    export log_dir=$log_dir

    # write config to $log_dir/config.txt
    echo "bits: $bits" > $log_dir/config.txt
    echo "topk_percent: $topk_percent" >> $log_dir/config.txt
    echo "threshold: $threshold" >> $log_dir/config.txt
    echo "kqa_threshold: $kqa_threshold" >> $log_dir/config.txt
    echo "kqa_topk_prop: $kqa_topk_prop" >> $log_dir/config.txt
    echo "profiling: $profiling" >> $log_dir/config.txt
    echo "oracle: $oracle" >> $log_dir/config.txt
    echo "cache_channel_selection: $cache_channel_selection" >> $log_dir/config.txt
    echo "baseline: $baseline" >> $log_dir/config.txt
    echo "log_dir: $log_dir" >> $log_dir/config.txt

    echo "bits: $bits, topk_percent: $topk_percent, threshold: $threshold, kqa_threshold: $kqa_threshold, kqa_topk_prop: $kqa_topk_prop, profiling: $profiling, oracle: $oracle, cache_channel_selection: $cache_channel_selection, baseline: $baseline, log_dir: $log_dir"

    if [ "$MODEL" == "swin2small" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swinv2/swinv2_small_patch4_window8_256.yaml --batch-size 32 --resume swinv2_small_patch4_window8_256.pth --data-path $imagenet_path

    elif [ "$MODEL" == "swin2tiny" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swinv2/swinv2_tiny_patch4_window16_256.yaml --batch-size 32 --resume swinv2_tiny_patch4_window16_256.pth --data-path $imagenet_path
    elif [ "$MODEL" == "swinsmall" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/swin_small_patch4_window7_224.yaml --resume swin_small_patch4_window7_224.pth --data-path $imagenet_path
    elif [ "$MODEL" == "swintiny" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --resume swin_tiny_patch4_window7_224.pth --data-path $imagenet_path
    elif [ "$MODEL" == "swinbase" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin/swin_base_patch4_window7_224.yaml --resume swin_base_patch4_window7_224.pth --data-path ~/imagenet/
    elif [ "$MODEL" == "swin2base" ]; then
        python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swinv2/swinv2_base_patch4_window16_256.yaml --resume swinv2_base_patch4_window16_256.pth --data-path ~/imagenet/
    fi
}

baseline=1
oracle=0
profiling=1
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swintiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swinsmall
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swinbase
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2tiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2small
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2base

baseline=0
oracle=1
profiling=0
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swintiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swinsmall
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swinbase
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2tiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2small
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 swin2base

baseline=0
oracle=0
profiling=0
bits=4
topk_percent=0.2
for cache in 1 0
do
for threshold in 0.01 0.05 0.1 0.15 0.2
do
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swintiny
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swinsmall
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swinbase
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swin2tiny
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swin2small
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache swin2base
done
done