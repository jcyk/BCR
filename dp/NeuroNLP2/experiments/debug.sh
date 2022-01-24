#!/usr/bin/env bash

for i in 1 2 3 4 5 6 7 8 9 10; do
set -e

data_dir=/home/jcykcai/nonono/BCR/dp/data2.2

OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test ${data_dir}/en_test.conllu \
 --model_path deepbiaf \
 --result_path out/deepbiaf \
 --kbest 1

###dropout sampling###
sample_seed=123
dropout=0.15 
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test ${data_dir}/en_test.conllu \
 --model_path deepbiaf \
 --result_path out/deepbiaf.dropout \
 --kbest 1  \
 --p_in ${dropout} --p_out ${dropout} --p_rnn ${dropout} \
 --seed ${sample_seed}

python replace.py out/deepbiaf.dropout.pred ${data_dir}/en_test.conllu 1


OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test out/deepbiaf.dropout.pred.post \
 --model_path deepbiaf \
 --result_path out/deepbiaf.dropout.pred.post



set -e

OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test ${data_dir}/en_test.conllu \
 --model_path stackptr \
 --result_path out/stackptr \
 --kbest 1



###dropout sampling###
sample_seed=123
dropout=0.15
OMP_NUM_THREADS=4 python -u parsing.py --mode parse --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test ${data_dir}/en_test.conllu \
 --model_path stackptr \
 --result_path out/stackptr.dropout \
 --kbest 1 \
 --p_in ${dropout} --p_out ${dropout} --p_rnn ${dropout} \
 --seed ${sample_seed}

python replace.py out/stackptr.dropout.pred ${data_dir}/en_test.conllu 1


OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/biaffine.json --num_epochs 400 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999995 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test out/stackptr.dropout.pred.post \
 --model_path stackptr \
 --result_path out/stackptr.dropout.pred.post.biaffine

OMP_NUM_THREADS=4 python -u parsing.py --mode score_reference --config configs/parsing/stackptr.json --num_epochs 600 --batch_size 32 \
 --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
 --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
 --word_embedding sskip --word_path ${data_dir}/sskip.eng.100.gz --char_embedding random \
 --train ${data_dir}/en_train.conllu \
 --dev ${data_dir}/en_dev.conllu \
 --test out/stackptr.dropout.pred.post \
 --model_path stackptr \
 --result_path out/stackptr.dropout.pred.post.stackptr

done
