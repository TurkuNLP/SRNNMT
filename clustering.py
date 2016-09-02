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
    
    
def cluster(sentences,vectors):
    k=AffinityPropagation()
    #k=MiniBatchKMeans(batch_size=200,n_clusters=30)
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

    fin_sentences=read_parsebank("pbv4_ud.part-00.gz",max_sent=10000)
    eng_sentences=["empty"]*len(fin_sentences) # fake english sentences for now, fix later!
#    print(fin_sentences,eng_sentences)
    fin_vectors,eng_vectors=test.vectorize(voc_name,fin_sentences,eng_sentences,model_name)   
    # now we have vectors for finnish sentences, build documents and cluster
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
    
    
    
    
    
