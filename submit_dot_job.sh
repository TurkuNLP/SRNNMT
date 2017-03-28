#!/bin/bash
#SBATCH -N 1
#SBATCH -p serial
#SBATCH -t 24:00:00
#SBATCH -o taito-logs/%j.out
#SBATCH -e taito-logs/%j.err
#SBATCH --mem=16000
#SBATCH -c 4


# files are for example fi.clusters430-439.gz, input should be clusters430-439
cluster=$1
echo $cluster
source $USERAPPL/activate_gpu.sh
time python dot.py --fi_fname /wrk/jmnybl/SRNNMT-tmp/vdata_ep100k/clusters/fi.$cluster --en_fname /wrk/jmnybl/SRNNMT-tmp/vdata_ep100k/clusters/en.$cluster --dictionary /wrk/jmnybl/SRNNMT-tmp/vdata_ep100k/dictionary_models/ep100k.lex --dict_vocabulary /wrk/jmnybl/SRNNMT-tmp/vdata_ep100k/dictionary_models/uniq.train.tokens.100K --outfile /wrk/jmnybl/SRNNMT-tmp/vdata_ep100k/results/$cluster.results > /homeappl/home/jmnybl/appl_taito/SRNNMT-new-pipelines/logs/$cluster.log 2>&1

