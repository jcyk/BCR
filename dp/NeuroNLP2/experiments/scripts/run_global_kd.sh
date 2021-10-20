#!/usr/bin/env bash
set -e

for method in deepbiaf stackptr; do
for seed in 123 456 789 555 666; do
echo "/efs/dengcai/dp/checkpoints/gkd.stackptr.$method$seed.1"
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/checkpoints/out/train.$method$seed.pred.post" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/gkd.stackptr.$method$seed.1" \
 --result_path "/efs/dengcai/dp/out/stackptr.$method$seed"


OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/efs/dengcai/dp/checkpoints/out/train.$method$seed.pred.post" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/gkd.deepbiaf.$method$seed.1" \
 --result_path "/efs/dengcai/dp/out/deepbiaf.$method$seed"

done
done