import os
import argparse

def process_logs(dir_path, output_dir, prefix):

    result_map = {}
    map_key = None
    oracle = False
    with open(os.path.join(output_dir, prefix+"_summary.csv"), "a") as summary_file:
        summary_file.write("bits,topk_percent,threshold,profiling,oracle,cache_channel_selection,baseline,log_dir,result,percent_quantized,precision,recall,channel_selection_stability\n")
        for log_dirs in os.listdir(dir_path):
            full_path = os.path.join(dir_path, log_dirs)
            if log_dirs == prefix+"_summary.csv" or log_dirs == "summary.csv":
                continue
            print(f"Processing {full_path}")
            files = os.listdir(full_path)
            if "results.csv" not in files:
                print(f"Skipping {full_path}, no results.csv")
                if "config.txt" in files:
                    with open(os.path.join(full_path, "config.txt"), "r") as config_file:
                        config = config_file.read()
                        print(f"Config:\n{config}")
                continue
            if "config.txt" in files:
                with open(os.path.join(full_path, "config.txt"), "r") as config_file:
                    config_lines = config_file.readlines()
                    config_dict = {}
                    for line in config_lines:
                        if ":" in line:
                            key, value = line.split(":")
                            config_dict[key.strip()] = value.strip()
                    summary_file.write(f"{config_dict.get('bits', '')},{config_dict.get('topk_percent', '')},{config_dict.get('threshold', '')},{config_dict.get('profiling', '')},{config_dict.get('oracle', '')},{config_dict.get('cache_channel_selection', '')},{config_dict.get('baseline', '')},{log_dirs},")
                    if config_dict.get('baseline', '') == "1":
                        map_key = 0.0
                    elif config_dict.get('baseline', '') == "0" and config_dict.get('oracle', '') == "0" and config_dict.get('cache_channel_selection', '') == "1":
                        map_key = float(config_dict.get('threshold', ''))
                    else:
                        map_key = None
                    if config_dict.get('oracle', '') == "1":
                        oracle = True
                    else:
                        oracle = False
                    if config_dict.get('profiling', '') == "0" and config_dict.get('baseline', '') == "0" and config_dict.get('oracle', '') == "0" and config_dict.get('cache_channel_selection', '') == "1" and config_dict.get('threshold', '') == "0.01":
                        channel_selection_stability = True
                    else:
                        channel_selection_stability = False
            if "results.csv" in files:
                with open(os.path.join(full_path, "results.csv"), "r") as results_file:
                    result_lines = results_file.readlines()
                    if len(result_lines) > 1:
                        print("Too many lines in results.csv, taking last one")
                    elif len(result_lines) == 0:
                        print("No lines in results.csv, skipping")
                        continue
                    last_line = result_lines[-1].strip()
                    summary_file.write(f"{last_line}")
                    if "," not in last_line:
                        summary_file.write(",")
                    if map_key is not None:
                        # result_map[map_key] = f"{float(last_line.split(",")[0]):.1f}"
                        acc = float(last_line.split(",")[0])
                        my_str = f"{acc:.1f}"
                        result_map[map_key] = my_str
                        if map_key == 0.0:
                            result_map[map_key] += " / 0"
                    if oracle:
                        acc = float(last_line.split(",")[0])
                        my_str = f"{acc:.1f}"
                        print(f"{prefix}: {my_str} /", end=" ")

            if "fp_channel_selection.csv" in files:
                with open(os.path.join(full_path, "fp_channel_selection.csv"), "r") as fp_file:
                    fp_lines = fp_file.readlines()
                    num_lines = len(fp_lines)
                    precision_sum = 0.0
                    recall_sum = 0.0
                    fp_channels_sum = 0
                    total_channels_sum = 0

                    for line in fp_lines:
                        line = line.strip()
                        line = line.split(",")
                        precision_sum += float(line[1])
                        recall_sum += float(line[2])
                        fp_channels_sum += int(line[3])
                        total_channels_sum += int(line[4])
                    precision = precision_sum / num_lines
                    recall = recall_sum / num_lines
                    percent_quantized = 1-(fp_channels_sum / total_channels_sum if total_channels_sum > 0 else 0)
                    summary_file.write(f"{percent_quantized},{precision},{recall},")
                    if map_key is not None:
                        result_map[map_key] += f" / {percent_quantized * 100:.1f}"
            if "channel_selection_stability.csv" in files:
                with open(os.path.join(full_path, "channel_selection_stability.csv"), "r") as stability_file:
                    stability_lines = stability_file.readlines()
                    num_lines = len(stability_lines)
                    stability_sum = 0.0

                    for line in stability_lines:
                        line = line.strip()
                        line = line.split(",")[1]
                        stability_sum += float(line)
                    stability = stability_sum / num_lines
                    summary_file.write(f"{stability},")
                    if channel_selection_stability:
                        print(f"{prefix}: {stability*100:.1f}%")
            if "oracle_quantization.csv" in files:
                with open(os.path.join(full_path, "oracle_quantization.csv"), "r") as oracle_file:
                    oracle_lines = oracle_file.readlines()
                    neg_el_count_sum = 0
                    el_count_sum = 0

                    for line in oracle_lines:
                        line = line.strip()
                        line = line.split(",")
                        neg_el_count_sum += int(line[1])
                        el_count_sum += int(line[2])
                    neg_el_prop = neg_el_count_sum / el_count_sum if el_count_sum > 0 else 0
                    summary_file.write(f"{neg_el_prop},")
                print(f"{neg_el_prop*100:.1f}")

            
            summary_file.write("\n")
    print(f"Result Map for {prefix}:")
    for key in sorted(result_map.keys()):
        print(f"{result_map[key]}")
    for key in sorted(result_map.keys()):
        print(f"{key}: {result_map[key]}")

# get log path from command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="logs", help="Path to the log directory")
args = parser.parse_args()

for dir_path in os.listdir(args.log_dir):
    full_path = os.path.join(args.log_dir, dir_path)
    if os.path.isdir(full_path):
        process_logs(full_path, full_path, dir_path)