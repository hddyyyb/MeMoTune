CUDA_VISIBLE_DEVICES=1 python test_gsm8k.py \
    --model_name_or_path ../lowbit_file/Llama-2-7b-hf-4bit-64rank/ \
    --adapter_name_or_path ./output/train_64rank_k2/file/Llama-2-7b-hf-4bit-64rank/ep_4/lr_0.0003/seed_17/ \
    --batch_size 16 \
    --scale 700 \
    --K 2 