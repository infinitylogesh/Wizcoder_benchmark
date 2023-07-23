#!/usr/bin/env bash
set -e

inference_engine=("hf" "hf_pipeline" "tgi" "vllm")
batch_size=(64 128)
num_tokens=(256 512)
input_scale_factor=(1 100)

# BS - 64 & Inference engine - hf
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine hf --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine hf --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine hf --input_scale_factor 100
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine hf --input_scale_factor 100

# BS - 64 & Inference engine - hf_pipeline
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine hf_pipeline --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine hf_pipeline --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine hf_pipeline --input_scale_factor 100
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine hf_pipeline --input_scale_factor 100

# BS - 64 & Inference engine - tgi
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine tgi --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine tgi --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine tgi --input_scale_factor 100
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine tgi --input_scale_factor 100

# BS - 64 & Inference engine - vllm
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine vllm --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine vllm --input_scale_factor 1
python3 main.py --batch_size 64 --num_tokens 256 --inference_engine vllm --input_scale_factor 100
python3 main.py --batch_size 64 --num_tokens 512 --inference_engine vllm --input_scale_factor 100

# BS - 128 & Inference engine - hf
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine hf --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine hf --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine hf --input_scale_factor 100
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine hf --input_scale_factor 100

# BS - 128 & Inference engine - hf_pipeline
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine hf_pipeline --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine hf_pipeline --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine hf_pipeline --input_scale_factor 100
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine hf_pipeline --input_scale_factor 100

# BS - 128 & Inference engine - tgi
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine tgi --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine tgi --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine tgi --input_scale_factor 100
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine tgi --input_scale_factor 100

# BS - 128 & Inference engine - vllm
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine vllm --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine vllm --input_scale_factor 1
python3 main.py --batch_size 128 --num_tokens 256 --inference_engine vllm --input_scale_factor 100
python3 main.py --batch_size 128 --num_tokens 512 --inference_engine vllm --input_scale_factor 100