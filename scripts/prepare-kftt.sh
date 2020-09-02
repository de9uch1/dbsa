#!/bin/bash
# Copyright (c) Hiroyuki Deguchi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

function not_exists() {
    ! [[ -d $TOOLS_DIR/$1 ]] && ! [[ -f $1 ]]
}
function corpora_not_exists() {
    ! [[ -d $orig ]] || ! [[ -f $orig/train.$src ]]
}

TOOLS_DIR=$(realpath ${TOOLS_DIR:-$HOME/.cache/nlp_tools})
mkdir -p $TOOLS_DIR

NUM_WORKERS=${NUM_WORKERS:-8}
BPE_TOKENS=16000

PARALLEL_SCRIPT=$TOOLS_DIR/parallel/bin/parallel
PARALLEL="$PARALLEL_SCRIPT --no-notice --pipe -j $NUM_WORKERS -k"
MOSES_SCRIPTS=$TOOLS_DIR/mosesdecoder/scripts
MOSES_TOKENIZER=$MOSES_SCRIPTS/tokenizer/tokenizer.perl
KYTEA_TOKENIZER=$TOOLS_DIR/kytea/bin/kytea
CLEAN=$TOOLS_DIR/dnlp/preprocess/clean-corpus-with-labels.py
BPEROOT=$TOOLS_DIR/fastBPE
FASTBPE=$BPEROOT/fast
EDA=$TOOLS_DIR/eda/eda
EDA_MODEL=$TOOLS_DIR/eda/fullmodel.etm
STANZA=$TOOLS_DIR/dnlp/tools/stanza_cli.py
BPE_DEP=$(dirname $0)/bpe_dependency.py

pushd $TOOLS_DIR >/dev/null
if not_exists $PARALLEL_SCRIPT; then
    echo 'Cloning GNU parallel github repository (for parallel execution)...'
    git clone https://git.savannah.gnu.org/git/parallel.git
    pushd parallel
    ./configure --prefix=$(pwd)
    make -j4
    make install
    popd
    if ! [[ -f $PARALLEL_SCRIPT ]]; then
        echo "GNU Parallel not successfully installed, abort."
        exit -1
    fi
fi
if not_exists mosesdecoder; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi
if not_exists $KYTEA_TOKENIZER; then
    echo 'Cloning KyTea github repository (for tokenization scripts)...'
    git clone https://github.com/neubig/kytea.git
    pushd kytea
    autoreconf -i
    ./configure --prefix=$(pwd)
    make -j4
    make install
    popd
    if not_exists $KYTEA_TOKENIZER; then
        echo "KyTea not successfully installed, abort."
        exit -1
    fi
fi
if not_exists $FASTBPE; then
    echo 'Cloning fastBPE repository (for BPE pre-processing)...'
    git clone https://github.com/glample/fastBPE.git
    pushd $BPEROOT
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    popd
    if not_exists $FASTBPE; then
        echo "fastBPE not successfully installed, abort."
        exit -1
    fi
fi
if not_exists dnlp; then
    echo 'Cloning dnlp repository (for some tools)...'
    git clone https://github.com/de9uch1/dnlp.git
fi
if not_exists $EDA; then
    echo 'Installing EDA (for Japanese dependency parsing)...'
    curl -sL "http://www.ar.media.kyoto-u.ac.jp/tool/EDA/downloads/eda-0.3.5.tar.gz" | tar xz
    mv eda-0.3.5 eda
    pushd eda
    make eda
    curl -sL "http://www.ar.media.kyoto-u.ac.jp/tool/EDA/downloads/20170713_fullmodel.tar.gz" | tar xz
    popd
    if not_exists $EDA; then
        echo "EDA not successfully installed, abort."
        exit -1
    fi
fi
popd >/dev/null

src=ja
tgt=en
OUTDIR=kftt
prep=$OUTDIR
tmp=$prep/tmp
orig=$prep/orig

mkdir -p $prep $orig $tmp
if corpora_not_exists; then
    pushd $orig
    curl -sL http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz | tar xz
    for split in train dev test; do
        for l in ja en; do
            mv kftt-data-1.0/data/orig/kyoto-$split.$l $split.$l
            if [[ $split = "dev" ]]; then
                mv $split.$l valid.$l
            fi
        done
    done
    rm -r kftt-data-1.0
    popd
    if corpora_not_exists; then
        echo "Corpora not successfully downloaded, abort."
        rm -r $orig
        exit -1
    fi
fi

echo "removing noise sentences..."
paste $orig/train.$src $orig/train.$tgt | \
    grep -v '（　〃　）' \
         > $tmp/train.$src-$tgt.tsv
cut -f1 $tmp/train.$src-$tgt.tsv > $tmp/train.$src
cut -f2 $tmp/train.$src-$tgt.tsv > $tmp/train.$tgt

for split in valid test; do
    for l in $src $tgt; do
        f=$split.$l
        cp $orig/$f $tmp/$f
    done
done

echo "tokenizing sentences in Japanese..."
for split in train valid test; do
    cat $tmp/$split.ja | \
        $PARALLEL $KYTEA_TOKENIZER | \
        tee $tmp/$split.ja.kytea | \
        $PARALLEL $KYTEA_TOKENIZER -in full -out tok | \
        perl -pe 's/　/ /g;' | \
        perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' \
             > $tmp/$split.ja.tok
    mv $tmp/$split.ja.tok $tmp/$split.ja
done

echo "tokenizing sentences in English..."
for split in train valid test; do
    cat $tmp/$split.en | \
        perl $MOSES_TOKENIZER -threads $NUM_WORKERS -l en \
             > $tmp/$split.en.tok
    mv $tmp/$split.en.tok $tmp/$split.en
done

echo "cleaning corpus..."
python $CLEAN --ratio 2.0 -l ja.kytea $tmp/train $src $tgt $tmp/train.clean 1 100
for L in $src $tgt; do
    cp $tmp/test.$L $prep/ref.$L
    mv $tmp/train.clean.$L $tmp/train.$L
done
mv $tmp/train.clean.ja.kytea $tmp/train.ja.kytea

echo "learn BPE on $tmp/train.$src, $tmp/train.$tgt..."
BPE_CODE=$prep/code
for L in $src $tgt; do
    f=train.$L
    c=$BPE_CODE.$L
    $FASTBPE learnbpe $BPE_TOKENS $tmp/$f > $c
    $FASTBPE applybpe $tmp/bpe.$f $tmp/$f $c
    $FASTBPE getvocab $tmp/bpe.$f > $prep/vocab.$L
done

for L in $src $tgt; do
    BPE_VOCAB=$prep/vocab.$L
    for f in train.$L valid.$L test.$L; do
        echo "apply BPE to ${f}..."
        $FASTBPE applybpe $prep/$f $tmp/$f $BPE_CODE.$L $BPE_VOCAB
    done
done

echo "parse dependency trees..."
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
        cut -d ' ' -f 7 | \
        perl -pe 's/-1/0/g; s/([0-9])\n/$1 /g' | \
        perl -pe 's/ $//g' \
             > $tmp/$f.dep
    python $BPE_DEP -t $prep/$f < $tmp/$f.dep > $prep/$f.dep
done
for split in train valid test; do
    f=$split.en
    python $STANZA --depparse -l en --batch-size 20000 < $tmp/$f > $tmp/$f.dep
    python $BPE_DEP -t $prep/$f < $tmp/$f.dep > $prep/$f.dep
done
