#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

BPEROOT=subword_nmt
BPE_TOKENS=100000

dataset=$1
prep=$dataset/raw
TRAIN=$prep/corpus_total.txt
BPE_CODE=$prep/code

src=src
tgt=tgt

output=$dataset/bpe-output

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $prep/$f > $output/$f
    done
done
