#from keras.models import Sequential, Graph, Model, model_from_json
#from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten
#from keras.layers.recurrent import GRU
#from keras.callbacks import Callback,ModelCheckpoint
#from keras.layers.embeddings import Embedding
import numpy as np
import sys
import math
import json
import gzip
import numpy as np

import data_dense
import test
import conllutil3 as cu
from sklearn.cluster import MiniBatchKMeans,AffinityPropagation


def read_parsebank(fname,max_sent=10000):

    sentences=[]
    counter=0
    for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
        if 5<=len(sent)<=30:
            txt=" ".join(line[cu.FORM] for line in sent)
            sentences.append(txt)
            counter+=1
        if counter==max_sent:
            break
    return sentences
    
def read_txt_stdin():

    sentences=[]
    counter=0
    for line in sys.stdin:
        sentences.append(line.strip())
    return sentences
    
    
def cluster(sentences,vectors):
#    k=AffinityPropagation()
    k=MiniBatchKMeans(batch_size=200,n_clusters=10)
    distances=k.fit_predict(vectors)
    d={} # key:label  value:listofitems
    for sent,label in zip(sentences,k.labels_):
        d.setdefault(label,[]).append(sent)
    for key, values in d.items():
        print("cluster:",key)
        for sent in values:
            print(sent)
        print()

    
    
def main(voc_name,model_name):

    fin_sentences=read_txt_stdin()
#    eng_sentences=[s for s in read_eng_parsebank("/home/jmnybl/git_checkout/SRNNMT/EN-COW/",max_sent=100)]
    
    eng_sentences=["empty"]*len(fin_sentences) # english input is not used
    
    
    print(len(fin_sentences),len(eng_sentences))
    fin_vectors,eng_vectors=test.vectorize(voc_name,fin_sentences,eng_sentences,model_name)
      
    cluster(fin_sentences,fin_vectors)
    
    
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give keras model name')
    #g.add_argument('--cutoff', type=int, default=2, help='Frequency threshold, how many times an ngram must occur to be included? (default %(default)d)')
    g.add_argument('-v', '--vocabulary', type=str, help='Give keras vocabulary file')
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)
    
    main(args.vocabulary,args.model)
    
    
    
    
    
