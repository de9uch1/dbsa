#!/bin/bash
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

function not_exists() {
    ! ( [[ -d $TOOLS_DIR/$1 ]] || [[ -f $1 ]] )
}
function corpora_not_exists() {
    ! [[ -d $orig ]] || ! [[ -f $orig/train.$src ]]
}

ASPEC_JE=${ASPEC_JE:-/path/to/ASPEC/ASPEC-JE}      # replace with your correct path
if ! [[ -f $ASPEC_JE/train/train-1.txt ]]; then
    cat << __EOT__
\$ASPEC_JE ($ASPEC_JE) is not the correct path.
Please set 'ASPEC_JE' to the correct path.

Example:
    When ASPEC-JE is given with the following path,
    you must set ASPEC_JE as follows:

    $ export ASPEC_JE="/path/to/ASPEC/ASPEC-JE"

    /path/to/ASPEC/ASPEC-JE
    ├── dev
    │   └── dev.txt
    ├── devtest
    │   └── devtest.txt
    ├── README
    ├── README-j
    ├── test
    │   └── test.txt
    └── train
        ├── train-1.txt
        ├── train-2.txt
        └── train-3.txt
__EOT__
    exit -1
fi

TOOLS_DIR=$(realpath ${TOOLS_DIR:-$HOME/.cache/nlp_tools})
mkdir -p $TOOLS_DIR

NUM_WORKERS=${NUM_WORKERS:-8}
NUM_WORKERS_GPU=${NUM_WORKERS_GPU:-2}
BPE_TOKENS=16000

PARALLEL_SCRIPT=$TOOLS_DIR/parallel/bin/parallel
PARALLEL="$PARALLEL_SCRIPT --no-notice --pipe -j $NUM_WORKERS -k"
PARALLEL_GPU="$PARALLEL_SCRIPT --no-notice --pipe -j $NUM_WORKERS_GPU -k"
MOSES_SCRIPTS=$TOOLS_DIR/mosesdecoder/scripts
MOSES_TOKENIZER=$MOSES_SCRIPTS/tokenizer/tokenizer.perl
KYTEA_TOKENIZER=$TOOLS_DIR/kytea/bin/kytea
NORM_PUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
Z2H=$TOOLS_DIR/dnlp/wat/z2h-utf8.perl
CLEAN=$TOOLS_DIR/dnlp/preprocess/clean-corpus-with-labels.py
BPEROOT=$TOOLS_DIR/fastBPE
FASTBPE=$BPEROOT/fast
EDA=$TOOLS_DIR/eda/eda
EDA_MODEL=$TOOLS_DIR/eda/fullmodel.etm
CONLLU2HEADS=$TOOLS_DIR/dnlp/tools/conllu2heads.py
STANZA=$TOOLS_DIR/dnlp/tools/stanza_cli.py
BPE_DEP=$(dirname $0)/bpe_dependency.py

pushd $TOOLS_DIR >/dev/null
if not_exists $PARALLEL_SCRIPT || \
        not_exists mosesdecoder || \
        not_exists $KYTEA_TOKENIZER || \
        not_exists $FASTBPE || \
        not_exists dnlp || \
        not_exists $EDA; then
    echo "===> Some tools are not installed, start installing..."
    script_mode='install'
fi

if not_exists $PARALLEL_SCRIPT; then
    echo '====> Installing GNU parallel (for parallel execution)...'
    git clone https://git.savannah.gnu.org/git/parallel.git
    pushd parallel
    ./configure --prefix=$(pwd)
    make -j4
    make install
    popd
    if ! [[ -f $PARALLEL_SCRIPT ]]; then
        echo "!!! GNU Parallel not successfully installed, abort."
        exit -1
    fi
    echo 'Done.'
fi
if not_exists mosesdecoder; then
    echo '====> Installing Moses (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
    echo 'Done.'
fi
if not_exists $KYTEA_TOKENIZER; then
    echo '====> Installing KyTea (for tokenization scripts)...'
    git clone https://github.com/neubig/kytea.git
    pushd kytea
    autoreconf -i
    ./configure --prefix=$(pwd)
    make -j4
    make install
    popd
    if not_exists $KYTEA_TOKENIZER; then
        echo "!!! KyTea not successfully installed, abort."
        exit -1
    fi
    echo 'Done.'
fi
if not_exists $FASTBPE; then
    echo '====> Installing fastBPE repository (for BPE pre-processing)...'
    git clone https://github.com/glample/fastBPE.git
    pushd $BPEROOT
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    popd
    if not_exists $FASTBPE; then
        echo "!!! fastBPE not successfully installed, abort."
        exit -1
    fi
    echo 'Done.'
fi
if not_exists dnlp; then
    echo '====> Installing dnlp (for some tools)...'
    git clone https://github.com/de9uch1/dnlp.git
    echo 'Done.'
fi
if not_exists $EDA; then
    echo '====> Installing EDA (for Japanese dependency parsing)...'
    curl -sL "http://www.ar.media.kyoto-u.ac.jp/tool/EDA/downloads/eda-0.3.5.tar.gz" | tar xz
    mv eda-0.3.5 eda
    pushd eda
    make eda
    curl -sL "http://www.ar.media.kyoto-u.ac.jp/tool/EDA/downloads/20170713_fullmodel.tar.gz" | tar xz
    popd
    if not_exists $EDA; then
        echo "!!! EDA not successfully installed, abort."
        exit -1
    fi
    echo 'Done.'
fi
if [[ $script_mode = 'install' ]]; then
    echo '===> Installation is complete!'
fi
popd >/dev/null

src=ja
tgt=en
train_size=1500000
OUTDIR=aspec_ja_en
prep=$OUTDIR
tmp=$prep/tmp
orig=$prep/orig
valid_set=dev

mkdir -p $prep $orig $tmp
cat $ASPEC_JE/train/train-{1,2,3}.txt | head -n $train_size > $orig/train.txt
cp $ASPEC_JE/dev/dev.txt $orig/dev.txt
cp $ASPEC_JE/devtest/devtest.txt $orig/devtest.txt
cp $ASPEC_JE/test/test.txt $orig/test.txt

set -e
echo "===> Start preprocessing"
echo "====> Extracting sentences..."
for split in dev devtest test; do
    perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < $orig/$split.txt > $orig/$split.ja
    perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $orig/$split.txt > $orig/$split.en
done
for split in train; do
    perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $orig/$split.txt > $orig/$split.ja
    perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < $orig/$split.txt > $orig/$split.en
done
echo "Done."

echo "====> Removing date expressions at EOS in Japanese in the training and development data to reduce noise..."
for split in train dev devtest; do
    mv $orig/$split.ja $orig/$split.ja.org
    cat $orig/$split.ja.org | perl -C -pe 'use utf8; s/(.)［[０-９．]+］$/$1/;' > $orig/$split.ja
    rm $orig/$split.ja.org
done
echo "Done."

pushd $orig >/dev/null
for l in $src $tgt; do
    ln -sf $valid_set.$l valid.$l
done
popd >/dev/null

echo "====> Tokenizing sentences in Japanese..."
for split in train valid test; do
    cat $orig/$split.ja | \
        perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' | \
        $PARALLEL $KYTEA_TOKENIZER | \
        tee $tmp/$split.ja.kytea | \
        $PARALLEL $KYTEA_TOKENIZER -in full -out tok | \
        perl -C -pe 'use utf8; s/　/ /g;' | \
        perl -C -pe 'use utf8; s/^ +//; s/ +$//; s/ +/ /g;' \
             > $tmp/$split.ja
done
echo "Done."

echo "====> Tokenizing sentences in English..."
for split in train valid test; do
    cat $orig/$split.en | \
        perl -C $Z2H | \
        perl -C $NORM_PUNC -l en | \
        perl -C $MOSES_TOKENIZER -threads $NUM_WORKERS -l en -a -no-escape 2>/dev/null | \
        perl -C -pe 'use utf8; s/^ +//; s/ +$//; s/ +/ /g;' \
             > $tmp/$split.en
done
echo "Done."

echo "====> Cleaning the corpus..."
python $CLEAN --ratio 2.0 -l ja.kytea $tmp/train $src $tgt $tmp/train.clean 1 100
for L in $src $tgt; do
    mv $tmp/train.clean.$L $tmp/train.$L
    cp $tmp/test.$L $prep/ref.$L
done
mv $tmp/train.clean.ja.kytea $tmp/train.ja.kytea
echo "Done."

echo "====> Learn BPE on $tmp/train.$src, $tmp/train.$tgt..."
BPE_CODE=$prep/code
for l in $src $tgt; do
    train=train.$l
    $FASTBPE learnbpe $BPE_TOKENS $tmp/$train > $BPE_CODE.$l
    $FASTBPE applybpe $tmp/bpe$BPE_TOKENS.$train $tmp/$train $BPE_CODE.$l
    $FASTBPE getvocab $tmp/bpe$BPE_TOKENS.$train > $prep/vocab.$l
done
echo "Done."

echo "====> Apply BPE..."
for l in $src $tgt; do
    BPE_VOCAB=$prep/vocab.$l
    for split in train valid test; do
        f=$split.$l
        echo "apply BPE to $f..."
        $FASTBPE applybpe $prep/$f $tmp/$f $BPE_CODE.$l $prep/vocab.$l
    done
done
echo "Done."

echo "====> Parsing dependency structures in Japanese..."
for split in train valid test; do
    f=$split.ja
    cat $tmp/$f.kytea | \
        perl -pe "s@　/空白/＿@@g" | \
        perl -pe "s@　/空白/くうはく@@g" | \
        perl -pe "s@　　/補助記号/＿＿@@g" | \
        perl -pe "s@　/@/@g" | \
        perl -pe "s@ 　@ @g" | \
        perl -pe "s@ +@ @g" \
             > $tmp/$f.kytea.clean
    mv $tmp/$f.kytea.clean $tmp/$f.kytea
    cat $tmp/$f.kytea | \
        $PARALLEL $EDA -i kytea -o conll -m $EDA_MODEL | \
        $CONLLU2HEADS --split-fwspace \
                      > $tmp/$f.dep
    python $BPE_DEP -t $prep/$f < $tmp/$f.dep > $prep/$f.dep
done
echo "Done."

echo "====> Parsing dependency structures in English..."
for split in train valid test; do
    f=$split.en
    cat $tmp/$f | \
        python $STANZA --depparse -l en --batch-size 10000 > $tmp/$f.dep
    python $BPE_DEP -t $prep/$f < $tmp/$f.dep > $prep/$f.dep
done
echo "Done."

echo "===> Preprocessing is all complete!"
