export TASK_NAME=cola
export INT_BIT=4
export LR=5e-5
export Seeds=0
CUDA_VISIBLE_DEVICES=0 python train_glue_MeMoTune.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 60 \
  --learning_rate $LR \
  --output_dir ./output/cola7 \
  --int_bit $INT_BIT \
  --scale 800 \
  --K 1 \
  --weight_decay 0 \
  --seed $Seeds \
  --loftq \
  --reduced_rank 32 \
  --decomposed_pretrained_ckpt_path decomposed_pretrained_ckpt_path \
  --quant_embedding \
  --gradient_accumulation_steps 2 \
  --quant_method normal_float \
  --num_iter 5 \
  --decompose
