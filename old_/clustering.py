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

from big_run_vectorize import read_fin_parsebank, read_eng_parsebank

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
    
    
def cluster(sentences,vectors,your_labels=[]):
    k=AffinityPropagation()
#    k=MiniBatchKMeans(batch_size=200,n_clusters=10)
    distances=k.fit_predict(vectors)
    d={} # key:label  value:listofitems
    if not your_labels:
        your_labels=[""]*len(sentences)
    for sent,label,your_label in zip(sentences,k.labels_,your_labels):
        d.setdefault(label,[]).append((sent,your_label))
    for key, values in d.items():
        print("cluster:",key)
        for sent,your_label in values:
            print(sent,your_label)
        print()

    
    
def main(voc_name,model_name):

#    fin_sentences=[s for s in read_fin_parsebank("parsebank/",max_sent=100)]
#    eng_sentences=[s for s in read_eng_parsebank("/home/jmnybl/git_checkout/SRNNMT/EN-COW/",max_sent=100)]
    
    eng_sentences=[]
    labels=[]
    for line in open("farmeh_relation_data/train_data_clean"):
        line=line.strip()
        label,sent=line.split("\t")
        labels.append(label)
        eng_sentences.append(sent)
    
    fin_sentences=["empty"]*len(eng_sentences)
    
    
    print(len(fin_sentences),len(eng_sentences))
    fin_vectors,eng_vectors=test.vectorize(voc_name,fin_sentences,eng_sentences,model_name)   
    # now we have vectors for finnish and english sentences, build documents and cluster
#    vectors=np.concatenate((fin_vectors,eng_vectors),axis=0)
#    sentences=fin_sentences+eng_sentences
#    sentences_shuffled=[]
#    vectors_shuffled=np.empty((len(sentences),150),dtype=np.float32)
    
#    from random import shuffle
#    index_shuf=[ i for i in range(len(sentences))]
#    shuffle(index_shuf)
#    for i,idx in enumerate(index_shuf):
#        sentences_shuffled.append(sentences[idx])
#        vectors_shuffled[i]=vectors[idx]
    
    
#    cluster(sentences_shuffled,vectors_shuffled)
    cluster(eng_sentences,eng_vectors,labels)
    
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
    
    
    
    
    
