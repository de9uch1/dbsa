Dependency-Based Self-Attention for Transformer NMT (Deguchi et al., 2019)
##########################################################################

Installation
============

.. code:: bash

   git clone https://github.com/de9uch1/dbsa.git
   cd dbsa/
   pip install ./

Training a Transformer + DBSA model on ASPEC-JE
===============================================

1. Extract and preprocess the ASPEC-JE data (including dependency parsing)
--------------------------------------------------------------------------

.. code:: bash

   export NUM_WORKERS=8  # specify the number of CPUs
   bash scripts/prepare-aspec.sh

2. Preprocess the dataset
-------------------------

.. code:: bash

   # binarize the dataset
   fairseq-preprocess --source-lang ja --target-lang en \
       --trainpref aspec_ja_en/train \
       --validpref aspec_ja_en/valid \
       --testpref aspec_ja_en/test \
       --destdir data-bin/ \
       --workers $NUM_WORKERS

   # deploy the dependency labels
   for split in train valid test; do
       for l in ja en; do
           cp aspec_ja_en/$split.$l.dep data-bin/$split.dep.$l
       done
   done

3. Train a model
----------------

.. code:: bash

   fairseq-hydra-train \
       --config-dir dbsa/configs/ \
       --config-name transformer_dep \ 
       common.user_dir=dbsa/

4. Evaluate and generate the translations
-----------------------------------------

.. code:: bash

   fairseq-generate \
       data-bin --gen-subset test \
       --user-dir dbsa/ \
       --task translation_dep \
       --path checkpoints/checkpoint_last.pt \
       --max-len-a 1 --max-len-b 50 \
       --post-process \
       --beam 5 --nbest 1

-  You can also generate dependencies (BPE level)

.. code:: bash

   fairseq-dbsa-generate \
       data-bin --gen-subset test \
       --user-dir dbsa/ \
       --task translation_dep \
       --path checkpoints/checkpoint_last.pt \
       --max-len-a 1 --max-len-b 50 \
       --print-dependency \
       --beam 5 --nbest 1
