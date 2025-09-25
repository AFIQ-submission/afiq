export bits=4
export topk_percent=0
export threshold=0
export profiling=0
export oracle=0
export baseline=0
export kqa_threshold=1
export kqa_topk_prop=1
export cache_channel_selection=1
export log_dir="logs/awq_search/"
mkdir -p $log_dir

MODEL=opt-125m

python -m awq.entry --model_path /path/to/opt/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=opt-1.3b

python -m awq.entry --model_path /path/to/opt/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=opt-6.7b

python -m awq.entry --model_path /path/to/opt/$MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=qwen2.5-7b

python -m awq.entry --model_path /path/to/qwen/Qwen2.5-7B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=llama-7b

python -m awq.entry --model_path /path/to/llama/llama-7b/ \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=llama-2-7b

python -m awq.entry --model_path /path/to/llama/Llama-2-7b-hf/ \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

MODEL=llama3-8b

python -m awq.entry --model_path /path/to/llama/Meta-Llama-3-8B \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

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

    if [ "$MODEL" == "opt-125m" ] || [ "$MODEL" == "opt-1.3b" ] || [ "$MODEL" == "opt-6.7b" ]; then
        python -m awq.entry --model_path /path/to/opt/$MODEL \
            --tasks wikitext \
            --w_bit 4 --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w4-g128.pt \
            --q_backend fake
    elif [ "$MODEL" == "llama-7b" ]; then
        python -m awq.entry --model_path /path/to/llama/llama-7b/ \
            --tasks wikitext \
            --w_bit 4 --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w4-g128.pt \
            --q_backend fake
    elif [ "$MODEL" == "llama-2-7b" ]; then
        python -m awq.entry --model_path /path/to/llama/Llama-2-7b-hf/ \
            --tasks wikitext \
            --w_bit 4 --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w4-g128.pt \
            --q_backend fake
    elif [ "$MODEL" == "llama3-8b" ]; then
        python -m awq.entry --model_path /path/to/llama/Meta-Llama-3-8B \
            --tasks wikitext \
            --w_bit 4 --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w4-g128.pt \
            --q_backend fake
    elif [ "$MODEL" == "qwen2.5-7b" ]; then
        python -m awq.entry --model_path /path/to/qwen/Qwen2.5-7B \
            --tasks wikitext \
            --w_bit 4 --q_group_size 128 \
            --load_awq awq_cache/$MODEL-w4-g128.pt \
            --q_backend fake
    fi
}

baseline=1
oracle=0
profiling=0
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-125m
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-1.3b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-6.7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama-7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama-2-7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama3-8b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 qwen2.5-7b


baseline=0
oracle=1
profiling=0
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-125m
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-1.3b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 opt-6.7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama-7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama-2-7b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 llama3-8b
inner_loop 4 0.1 0.1 $profiling $oracle $baseline 1 qwen2.5-7b

baseline=0
oracle=0
profiling=0
bits=8
topk_percent=0.2
for cache in 1 0
do
for threshold in 0.01 0.05 0.1 0.15 0.2
do
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache opt-125m
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache opt-1.3b
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache opt-6.7b
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache llama-7b
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache llama-2-7b
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache llama3-8b
    inner_loop $bits $topk_percent $threshold $profiling $oracle $baseline $cache qwen2.5-7b
done
done
