#!/bin/bash
set -e
MODEL=$1
SEED=$2

### dropout sampling###
CKPT=/efs/dengcai/top/checkpoints
for dropout in 0.3; do
for sample_seed in 11 12 13 14 15 16 17 18 19 20; do
  DIR=${CKPT}/${MODEL}.seed${SEED}
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset test \
	--batch-size 128 --retain-dropout --beam 1 \
    --model-overrides "{'dropout': ${dropout}, 'attention-dropout': ${dropout}}" --seed ${sample_seed}> $DIR/test.parse.dropout${dropout}sample${sample_seed}.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/test.parse.dropout${dropout}sample${sample_seed}.unprocessed \
	--outfile /efs/dengcai/top/out/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.full.parse
    
   cp /efs/dengcai/top/out/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.full.parse.preprocessed.en /efs/dengcai/top/data/data/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.en
   cp /efs/dengcai/top/out/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.full.parse.preprocessed.parse /efs/dengcai/top/data/data/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.parse
   
   fairseq-preprocess --source-lang en --target-lang parse \
    --trainpref /efs/dengcai/top/data/data/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
    --validpref /efs/dengcai/top/data/data/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
    --testpref /efs/dengcai/top/data/data/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
    --destdir /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
    --srcdict /efs/dengcai/top/data/data/dict.en.txt \
    --tgtdict /efs/dengcai/top/data/data/dict.parse.txt  \
    --workers 20

    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/base.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.full.parse.base${OLD}.score
    done
    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.dropout${dropout}sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/large.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.dropout${dropout}sample${sample_seed}.full.parse.large${OLD}.score
    done
done
done

###top-k sampling###

CKPT=/efs/dengcai/top/checkpoints
for topk in 5; do
for sample_seed in 11 12 13 14 15 16 17 18 19 20; do
  DIR=${CKPT}/${MODEL}.seed${SEED}
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset test \
	--batch-size 128 --beam 1 --nbest 1 \
    --sampling --sampling-topk $topk --seed $sample_seed > $DIR/test.parse.topk${topk}.sample${sample_seed}.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/test.parse.topk${topk}.sample${sample_seed}.unprocessed \
	--outfile /efs/dengcai/top/out/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.parse
    
    
   cp /efs/dengcai/top/out/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.parse.preprocessed.en /efs/dengcai/top/data/data/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.en
   cp /efs/dengcai/top/out/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.parse.preprocessed.parse /efs/dengcai/top/data/data/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.parse
   
   fairseq-preprocess --source-lang en --target-lang parse \
    --trainpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
    --validpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
    --testpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
    --destdir /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
    --srcdict /efs/dengcai/top/data/data/dict.en.txt \
    --tgtdict /efs/dengcai/top/data/data/dict.parse.txt  \
    --workers 20

    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/base.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.base${OLD}.score
    done
    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topk${topk}.sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/large.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.topk${topk}.sample${sample_seed}.large${OLD}.score
    done
done
done


###top-p sampling###
for topp in 0.95; do
for sample_seed in 11 12 13 14 15 16 17 18 19 20; do
  DIR=${CKPT}/${MODEL}.seed${SEED}
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset test \
	--batch-size 128 --beam 1 --nbest 1 \
    --sampling --sampling-topp $topp --seed $sample_seed > $DIR/test.parse.topp${topp}.sample${sample_seed}.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/test.parse.topp${topp}.sample${sample_seed}.unprocessed \
	--outfile /efs/dengcai/top/out/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.parse
    
    
   cp /efs/dengcai/top/out/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.parse.preprocessed.en /efs/dengcai/top/data/data/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.en
   cp /efs/dengcai/top/out/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.parse.preprocessed.parse /efs/dengcai/top/data/data/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.parse
   
   fairseq-preprocess --source-lang en --target-lang parse \
    --trainpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
    --validpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
    --testpref /efs/dengcai/top/data/data/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
    --destdir /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
    --srcdict /efs/dengcai/top/data/data/dict.en.txt \
    --tgtdict /efs/dengcai/top/data/data/dict.parse.txt  \
    --workers 20

    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/base.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.base${OLD}.score
    done
    
    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.topp${topp}.sample${sample_seed} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/large.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.topp${topp}.sample${sample_seed}.large${OLD}.score
    done
done
done


exit 0

### KD ###
CKPT=/efs/dengcai/top/checkpoints
for MODEL in base large; do
for SEED in 123 456 789 555 666; do
  DIR=${CKPT}/${MODEL}.${MODEL}${SEED}.seed1
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset test \
	--batch-size 128 --beam 1 --nbest 1 > $DIR/test.parse.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/test.parse.unprocessed \
	--outfile /efs/dengcai/top/out/${MODEL}.${MODEL}${SEED}.seed1.parse
done
done

exit 0
### ensemble ###
DIR=/efs/dengcai/top/checkpoints/base.seed

NAME=5A
fairseq-generate /efs/dengcai/top/data/data-bin/top \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path ${DIR}123/checkpoint_best.pt:${DIR}456/checkpoint_best.pt:${DIR}789/checkpoint_best.pt:${DIR}555/checkpoint_best.pt:${DIR}666/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --batch-size 64 --beam 1 > seq2seq.base.${NAME}.parse.unprocessed

python process_gen_pg_file.py \
        --file seq2seq.base.${NAME}.parse.unprocessed \
        --outfile /efs/dengcai/top/out/seq2seq.base.${NAME}.parse

NAME=5B
fairseq-generate /efs/dengcai/top/data/data-bin/top \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path ${DIR}1/checkpoint_best.pt:${DIR}2/checkpoint_best.pt:${DIR}3/checkpoint_best.pt:${DIR}4/checkpoint_best.pt:${DIR}5/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --batch-size 64 --beam 1 > seq2seq.base.${NAME}.parse.unprocessed

python process_gen_pg_file.py \
        --file seq2seq.base.${NAME}.parse.unprocessed \
        --outfile /efs/dengcai/top/out/seq2seq.base.${NAME}.parse


NAME=5C
fairseq-generate /efs/dengcai/top/data/data-bin/top \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path ${DIR}6/checkpoint_best.pt:${DIR}7/checkpoint_best.pt:${DIR}8/checkpoint_best.pt:${DIR}9/checkpoint_best.pt:${DIR}10/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --batch-size 64 --beam 1 > seq2seq.base.${NAME}.parse.unprocessed

python process_gen_pg_file.py \
        --file seq2seq.base.${NAME}.parse.unprocessed \
        --outfile /efs/dengcai/top/out/seq2seq.base.${NAME}.parse

NAME=5D
fairseq-generate /efs/dengcai/top/data/data-bin/top \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path ${DIR}11/checkpoint_best.pt:${DIR}12/checkpoint_best.pt:${DIR}13/checkpoint_best.pt:${DIR}14/checkpoint_best.pt:${DIR}15/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --batch-size 64 --beam 1 > seq2seq.base.${NAME}.parse.unprocessed

python process_gen_pg_file.py \
        --file seq2seq.base.${NAME}.parse.unprocessed \
        --outfile /efs/dengcai/top/out/seq2seq.base.${NAME}.parse

NAME=5E
fairseq-generate /efs/dengcai/top/data/data-bin/top \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path ${DIR}16/checkpoint_best.pt:${DIR}17/checkpoint_best.pt:${DIR}18/checkpoint_best.pt:${DIR}19/checkpoint_best.pt:${DIR}20/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --batch-size 64 --beam 1 > seq2seq.base.${NAME}.parse.unprocessed

python process_gen_pg_file.py \
        --file seq2seq.base.${NAME}.parse.unprocessed \
        --outfile /efs/dengcai/top/out/seq2seq.base.${NAME}.parse
        

exit 0     

### beam search###
CKPT=/efs/dengcai/top/checkpoints
MODEL=large
SEED=$1
for beam in 2 4 8 10; do
  DIR=${CKPT}/${MODEL}.seed${SEED}
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset test \
	--batch-size 128 --beam $beam --nbest $beam >  $DIR/test.parse.b${beam}.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/test.parse.b${beam}.unprocessed \
	--outfile /efs/dengcai/top/out/${MODEL}${SEED}.b${beam}.full.parse
    
   cp /efs/dengcai/top/out/${MODEL}${SEED}.b${beam}.full.parse.preprocessed.en /efs/dengcai/top/data/data/${MODEL}${SEED}.b${beam}.en
   cp /efs/dengcai/top/out/${MODEL}${SEED}.b${beam}.full.parse.preprocessed.parse /efs/dengcai/top/data/data/${MODEL}${SEED}.b${beam}.parse
   
   fairseq-preprocess --source-lang en --target-lang parse \
    --trainpref /efs/dengcai/top/data/data/${MODEL}${SEED}.b${beam} \
    --validpref /efs/dengcai/top/data/data/${MODEL}${SEED}.b${beam} \
    --testpref /efs/dengcai/top/data/data/${MODEL}${SEED}.b${beam} \
    --destdir /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.b${beam} \
    --srcdict /efs/dengcai/top/data/data/dict.en.txt \
    --tgtdict /efs/dengcai/top/data/data/dict.parse.txt  \
    --workers 20

    for OLD in 123 456 789 555 666; do
       fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.b${beam} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/base.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.b${beam}.full.parse.base${OLD}.score
        fairseq-generate /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.b${beam} \
        --user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
        --path $CKPT/large.seed${OLD}/checkpoint_best.pt \
        --task translation_from_pretrained_roberta \
        --left-pad-source False \
        --gen-subset test \
        --score-reference \
        --batch-size 128 --beam 1 > /efs/dengcai/top/out/${MODEL}${SEED}.b${beam}.full.parse.large${OLD}.score
    done
done


exit 0

CKPT=/efs/dengcai/top/checkpoints
for MODEL in base large; do
for SEED in 123 456 789 555 666; do
  DIR=${CKPT}/${MODEL}.seed${SEED}
  fairseq-generate /efs/dengcai/top/data/data-bin/top \
	--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
	--path $DIR/checkpoint_best.pt \
	--task translation_from_pretrained_roberta \
	--left-pad-source False \
	--gen-subset train \
	--batch-size 128 --beam 1 > $DIR/train.parse.unprocessed

   python process_gen_pg_file.py \
	--file $DIR/train.parse.unprocessed \
	--outfile ${CKPT}/out/train.${MODEL}${SEED}.parse
done
done
