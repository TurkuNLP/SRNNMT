# SRNNMT
Sentence representation for translation finding

## Training NN models

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_conv.py --model_name test.model --src_train data/all.train.fi.tokenized --trg_train data/all.train.en.tokenized --src_devel data/all.dev.fi.tokenized --trg_devel data/all.dev.en.tokenized

## Building dictionaries

* TODO

## Vectorizing data

Read Finnish and English parsebank files, and filter data to keep only unique sentences which passes length and character filters, split the sentences into size-wise batches:

    python3 filter_uniq.py --inp /home/jmnybl/git_checkout/SRNNMT/parsebank --preprocessor conllu --outdir vdata_testing --out_prefix fi --max_sent 0
    python3 filter_uniq.py --inp /home/jmnybl/git_checkout/SRNNMT/EN-COW --preprocessor cow --outdir vdata_testing --out_prefix en  --max_sent 0

Read Finnish and English text files and turn these into dense vectors:

    for n in {5..30} ; do echo $n ; THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cnmem=0 python vectorize_dense.py -m test.model.37 -v test.model-vocab.pickle -l $n --data_dir vdata_testing --max_pairs 0 ; done
    
This expects _fi_ and _en_ prefixes, should be changed to _src_ and _trg_ (TODO).

## Cluster dense vectors

Calculate cluster centers based on a sample of the data (produces dir/lang.cluster.centers.pkl):

    python cluster_dense.py --dir vdata_testing --lang en --ratio 0.05 --clusters 1000

Times: ~48 min

Distribute data to cluster based files (produces dir/clusters/lang.clusters0-99):

    python data_to_clusters.py --dir vdata_testing --lang en --files 100 --clusters 1000
    python data_to_clusters.py --dir vdata_testing --lang fi --files 100 --clusters 1000
    
Times: English 304M sentences ~443min, Finnish 172M sentences ~246min

## Run dot product

Calculate sparse matrices, and run dense and sparse dot products inside clusters, slice data according to cluster ids and sentence lengths to keep things fast:

    python dot.py --fi_fname vdata_testing/clusters/fi.clusters430-439 --en_fname vdata_testing/clusters/en.clusters430-439 --dictionary europarl100K_exp/ep100k.lex --dict_vocabulary europarl100K_exp/uniq.train.tokens.100K --outfile results_testing/clusters430-439.results > clusters430-439.log 2>&1
    
Times: (clusters430-439 (smallest) = 23 min, clusters10-19 (biggest) = 233 min)
