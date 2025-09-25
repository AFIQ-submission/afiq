# afiq

## Installation
1. Install all dependencies of DeiT, Swin-Transformer, and llm-awq as per their repo's instructions. It is recommended to do this using venv for DeiT and Swin-Transformer, and conda for llm-awq.
2. For DeiT and Swin-Transformer, uninstall timm and install src/timm like so: `cd [deit,Swin-Transformer]; pip install -e ../timm`.
For llm-awq, uninstall transformers and install src/transformers-4.46.0m/ like so: `cd llm-awq; pip install -e ../transformers-4.46.0m/`
3. Run experiments with the `run_experiments.sh` script after replacing the path placeholders with your own
4. Process logs to gather data `python process_logs.py --log_dir [log_folder]`

