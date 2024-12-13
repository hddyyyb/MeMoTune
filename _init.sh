CUDA_VISIBLE_DEVICES=1
SAVE_DIR="../lowbit_file/"
python _init.py \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --token your_token_to_access_llama2_from_HuggingFace_Hub \
    --bits 4 \
    --iter 5 \
    --rank 64 \
    --save_dir $SAVE_DIR
