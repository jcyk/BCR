#!/usr/bin/env bash
set -e
data_dir=/home/jcykcai/nonono/BCR/dp/data2.2
for seed in 1; do
OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test ${data_dir}/en_test.conllu \
 --seed $seed \
 --model_path stackptr
done
