set -e
CKPT=/efs/dengcai/top/checkpoints
MODEL=$1
SEED=$2

for MODEL in base large; do
for SEED in 123 456 789 555 666; do
DIR=${CKPT}/${MODEL}.seed${SEED}
fairseq-generate /efs/dengcai/top/data/data-bin/top \
--user-dir /efs/dengcai/top/fairseq-01042021/examples/roberta_tops/ \
--path $DIR/checkpoint_best.pt \
--task translation_from_pretrained_roberta \
--left-pad-source False \
--gen-subset train \
--batch-size 128 --beam 1 --nbest 1 > $DIR/train.parse.unprocessed

python process_gen_pg_file.py \
--file $DIR/train.parse.unprocessed \
--outfile $DIR/train.parse
    
cp $DIR/train.parse.preprocessed.en /efs/dengcai/top/data/data/${MODEL}${SEED}.train.en
cp $DIR/train.parse.preprocessed.parse /efs/dengcai/top/data/data/${MODEL}${SEED}.train.parse

fairseq-preprocess --source-lang en --target-lang parse \
--trainpref /efs/dengcai/top/data/data/${MODEL}${SEED}.train \
--validpref /efs/dengcai/top/data/data/eval \
--testpref /efs/dengcai/top/data/data/test \
--destdir /efs/dengcai/top/data/data-bin/${MODEL}${SEED}.train \
--srcdict /efs/dengcai/top/data/data/dict.en.txt \
--tgtdict /efs/dengcai/top/data/data/dict.parse.txt  \
--workers 20

done
done