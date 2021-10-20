#!/usr/bin/env bash
set -e
for seed in $(seq 1 1 20); do
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr${seed}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}" \
 --kbest 1
done
exit 0

###ensemble##
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/checkpoints/stackptr123:/efs/dengcai/dp/checkpoints/stackptr456:/efs/dengcai/dp/checkpoints/stackptr789:/efs/dengcai/dp/checkpoints/stackptr555:/efs/dengcai/dp/checkpoints/stackptr666" \
 --result_path "/efs/dengcai/dp/out/stackptr5A"

OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr1:/efs/dengcai/dp/models/stackptr2:/efs/dengcai/dp/models/stackptr3:/efs/dengcai/dp/models/stackptr4:/efs/dengcai/dp/models/stackptr5" \
 --result_path "/efs/dengcai/dp/out/stackptr5B"
 
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr6:/efs/dengcai/dp/models/stackptr7:/efs/dengcai/dp/models/stackptr8:/efs/dengcai/dp/models/stackptr9:/efs/dengcai/dp/models/stackptr10" \
 --result_path "/efs/dengcai/dp/out/stackptr5C"

OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr11:/efs/dengcai/dp/models/stackptr12:/efs/dengcai/dp/models/stackptr13:/efs/dengcai/dp/models/stackptr14:/efs/dengcai/dp/models/stackptr15" \
 --result_path "/efs/dengcai/dp/out/stackptr5D"


OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr16:/efs/dengcai/dp/models/stackptr17:/efs/dengcai/dp/models/stackptr18:/efs/dengcai/dp/models/stackptr19:/efs/dengcai/dp/models/stackptr20" \
 --result_path "/efs/dengcai/dp/out/stackptr5E"
 
###beam search###
kbest=$1

for seed in 1 2 3 4 5; do
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr${seed}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.bs${kbest}" \
 --kbest ${kbest}

python replace.py /efs/dengcai/dp/out/stackptr${seed}.bs${kbest}.pred /efs/dengcai/dp/data2.2/en_test.conllu $kbest

for old in 123 456 789 555 666; do
OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/out/stackptr${seed}.bs${kbest}.pred.post" \
 --model_path "/efs/dengcai/dp/checkpoints/stackptr${old}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.bs${kbest}.stackptr${old}" \
 --kbest ${kbest}
 
OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/out/stackptr${seed}.bs${kbest}.pred.post" \
 --model_path "/efs/dengcai/dp/checkpoints/deepbiaf${old}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.bs${kbest}.deepbiaf${old}" \
 --kbest $kbest 
done
done

###dropout sampling###

sampl_seed=$1

for seed in 1 2 3 4 5; do
for dropout in 0.05 0.10 0.15 0.20 0.25 0.30 0.35; do
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/data2.2/en_test.conllu" \
 --model_path "/efs/dengcai/dp/models/stackptr${seed}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}" \
 --kbest 1 \
 --p_in ${dropout} --p_out ${dropout} --p_rnn ${dropout} \
 --seed ${sample_seed}

python replace.py /efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}.pred /efs/dengcai/dp/data2.2/en_test.conllu 1

for old in 123 456 789 555 666; do
OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}.pred.post" \
 --model_path "/efs/dengcai/dp/checkpoints/deepbiaf${old}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}.deepbiaf${old}"

OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path "/efs/dengcai/dp/sskip.eng.100.gz" --char_embedding random \
 --train "/efs/dengcai/dp/data2.2/en_train.conllu" \
 --dev "/efs/dengcai/dp/data2.2/en_dev.conllu" \
 --test "/efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}.pred.post" \
 --model_path "/efs/dengcai/dp/checkpoints/stackptr${old}" \
 --result_path "/efs/dengcai/dp/out/stackptr${seed}.dropout${dropout}sample${sample_seed}.stackptr${old}"
done
done
done