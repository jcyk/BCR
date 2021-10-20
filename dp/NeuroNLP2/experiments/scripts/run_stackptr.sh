#!/usr/bin/env bash
set -e
for seed in $(seq 50); do
OMP_NUM_THREADS=4 python -u parsing.py --mode train --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --seed $seed \
 --model_path "/home/ubuntu/dp/models/stackptr$seed/"

cp -r /home/ubuntu/dp/models/stackptr$seed /efs/dengcai/dp/models/stackptr$seed
done
