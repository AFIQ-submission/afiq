#! /bin/bash
imagenet_path=/path/to/imagenet

inner_loop() {
    local bits=$1
    local topk_percent=$2
    local threshold=$3
    local profiling=$4
    local oracle=$5
    local baseline=$6
    local size=$7

    uuid=$(uuidgen)
    log_dir=logs/$size/$uuid
    mkdir -p $log_dir
    export bits=$bits
    export topk_percent=$topk_percent
    export threshold=$threshold
    export kqa_threshold=1
    export kqa_topk_prop=1
    export profiling=$profiling
    export oracle=$oracle
    export cache_channel_selection=1
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
    if [ "$size" == "tiny" ]; then
        time python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --data-path $imagenet_path --model deit_tiny_patch16_224
    elif [ "$size" == "small" ]; then
        python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --model deit_small_patch16_224 --data-path $imagenet_path
    elif [ "$size" == "base" ]; then
        python main.py --eval --resume https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --data-path $imagenet_path
    fi
}

baseline=1
oracle=0
profiling=1
inner_loop 4 0.1 0.1 $profiling $oracle $baseline tiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline small
inner_loop 4 0.1 0.1 $profiling $oracle $baseline base

baseline=0
oracle=1
profiling=0
inner_loop 4 0.1 0.1 $profiling $oracle $baseline tiny
inner_loop 4 0.1 0.1 $profiling $oracle $baseline small
inner_loop 4 0.1 0.1 $profiling $oracle $baseline base

oracle=0
profiling=0
bits=4
for topk_percent in 0.01 0.05 0.1 0.15 0.2
do
for threshold in 0.01 0.05 0.1 0.15 0.2
do
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline tiny
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline small
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline base
done
done
