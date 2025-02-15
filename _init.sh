SAVE_DIR="../lowbit_file/"
python _init.py \
    --model_name_or_path yahma/llama-7b-hf \
    --token None \
    --bits 4 \
    --iter 5 \
    --rank 64 \
    --K 2 \
    --save_dir $SAVE_DIR
 
# Note:
# Replace 'None' in --token with your actual Hugging Face Hub token to download Llama2 models (7B or 13B) if required.
# Ensure that the save directory ($SAVE_DIR) exists and has write permissions.
