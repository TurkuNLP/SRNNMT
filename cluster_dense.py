import numpy as np
import sys
import math
import json
import gzip
import glob

from sklearn.cluster import MiniBatchKMeans

def sample(directory,lang,limit=0.05):
    # sample x percent of each length, bit clumsy, but works...
    sample=None
    for fname in glob.glob("{}/{}_len*.npy".format(directory,lang)):
        print(fname,file=sys.stderr)
        vectors=np.fromfile(fname,np.float32)
        vectors=vectors.reshape(int(len(vectors)/150),150)
        print(vectors.shape,file=sys.stderr)
        sampled=vectors[np.random.choice(vectors.shape[0], round(vectors.shape[0]*limit), replace=False),:]
        if isinstance(sample,type(None)):
            sample=sampled
        else:
            sample=np.concatenate((sample,sampled),axis=0)
        print("size",sample.shape,file=sys.stderr)
    return sample
    
def cluster(vectors,clusters=100):
    k=MiniBatchKMeans(batch_size=200,n_clusters=clusters)
    distances=k.fit_predict(vectors)
#    d={} # key:label  value:listofitems
#    for sent,label in zip(sentences,k.labels_):
#        d.setdefault(label,[]).append(sent)
#    total=0
#    for key, values in d.items():
#        print("cluster:",key,"#sentences:",len(values))
#        total+=len(values)
#        for sent in values:
#            print(sent)
#        print()
#    print("number of clusters:",clusters,"total:",total)

    
    
def main(directory,lang,limit=0.05,clusters=100):

#    print(limit)
    sampled_data=sample(directory,lang,limit=limit)
#    vectors=np.fromfile("vdata_ep100k/{L}_len{N}.npy".format(L=lang,N=sent_length),np.float32)
#    print(vectors.shape)
#    vectors=vectors.reshape(int(len(vectors)/150),150)
#    print(vectors.shape)
#    vectors=vectors[:limit,:]

    
#    sentences=[]
#    for i,s in enumerate(gzip.open("vdata_ep100k/{L}_len{N}.txt.gz".format(L=lang,N=sent_length),"rt",encoding="utf-8")):
#        sentences.append(s)
#        if i>=limit-1:
#            break
#    print(vectors.shape,len(sentences))

    cluster(sampled_data,clusters=clusters)
    
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
#    g.add_argument('--length', type=str, help='Sentence length, so we know which file to read')
    g.add_argument('--dir', type=str, help='Directory to read')
    g.add_argument('--lang', type=str, help='Language (fi or en), so we know which file to read')
    g.add_argument('--ratio', type=float, default=0.05, help='How much to sample from each length, default={n}'.format(n=0.05))
    g.add_argument('--clusters', type=int, default=100, help='Number of clusters, default={n}'.format(n=100))
    
    args = parser.parse_args()

    if args.dir==None or args.lang==None:
        parser.print_help()
        sys.exit(1)
    
    main(args.dir,args.lang,limit=args.ratio,clusters=args.clusters)
    
    
    
    
    
