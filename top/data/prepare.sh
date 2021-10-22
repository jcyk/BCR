

# require orig-modif-data: "We replaced the end-brackets with custom end-brackets corresponding to the intent or slot they close"
# roberta.base: tokenizer stuffs for roberta.base

set -e


python prepare_roberta_pointer.py --split train
python prepare_roberta_pointer.py --split eval
python prepare_roberta_pointer.py --split test --lines $inp.lines

cp roberta.base/dict.txt data/dict.en.txt
cat data/train.parse data/eval.parse data/test.parse |
  tr -s '[:space:]' '\n' |
  sort |
  uniq -c |
  sort -k1,1bnr -k2 |
  head -n 11304 |
  awk '{ print $2 " " $1 }' > data/dict.parse.txt.unprocessed

python format_vocab_pointer.py


fairseq-preprocess --source-lang en --target-lang parse \
    --trainpref data/train \
    --validpref data/eval \
    --testpref data/test \
    --destdir data-bin/rerank \
    --srcdict data/dict.en.txt \
    --tgtdict data/dict.parse.txt  \
    --workers 20
