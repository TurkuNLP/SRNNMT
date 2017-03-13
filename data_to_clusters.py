import numpy as np
import sys
import math
import json
import gzip
import glob
import pickle
import os

from sklearn.cluster import MiniBatchKMeans

def create_files(directory,lang,number_of_clusters,number_of_files=100):
    files={}
    if not os.path.exists("{}/clusters".format(directory)):
        print("Making directory {}/clusters".format(directory),file=sys.stderr)
        os.makedirs("{}/clusters".format(directory))
    step=int(math.ceil(number_of_clusters/number_of_files))
    cluster2file={}
    for i in range(0,number_of_clusters-1,step):
#        print(i,step,min(i+step-1,number_of_clusters-1))
        fname="{}/clusters/{}.clusters{}-{}.npy".format(directory,lang,i,min(i+step-1,number_of_clusters-1))
        print(fname,file=sys.stderr)
        files[fname]=open(fname,"wb")
        fname="{}/clusters/{}.clusters{}-{}.gz".format(directory,lang,i,min(i+step-1,number_of_clusters-1))
        print(fname,file=sys.stderr)
        files[fname]=gzip.open(fname,"wt",encoding="utf-8")
        fname="{}/clusters/{}.clusters{}-{}.meta".format(directory,lang,i,min(i+step-1,number_of_clusters-1))
        print(fname,file=sys.stderr)
        files[fname]=open(fname,"wb")
        
        for cluster in range(i,i+step):
            assert cluster not in cluster2file
            cluster2file[cluster]=fname
        
    return files,cluster2file
    
def load_cluster_model(directory,lang):
    with open("{}/{}.cluster.centers.pkl".format(directory,lang),"rb") as f:
        k=pickle.load(f)
    return k
    
def to_clusters(txt_fname,cluster_model,files,cluster2file):
    # metadata will be cluster id, sentence length, sentence id?
    # read data
    print(txt_fname,file=sys.stderr)
    v_fname=txt_fname.replace(".txt.gz",".npy")
    vectors=np.fromfile(v_fname,np.float32)
    vectors=vectors.reshape(int(len(vectors)/150),150)
    print("vectors:",vectors.shape,file=sys.stderr)
    sentences=[s for s in gzip.open(txt_fname,"rt",encoding="utf-8")]
    print("sentences:",len(sentences),file=sys.stderr)
    
    # predict
    print("Predicting clusters...",file=sys.stderr)
    labels=cluster_model.predict(vectors)
    print(labels[:100],file=sys.stderr)
    print("Saving data...",file=sys.stderr)
    for i,label in enumerate(labels):
        files[cluster2file[label].replace(".meta",".gz")].write(sentences[i])
        vectors[i,:].astype(np.float32).tofile(files[cluster2file[label].replace(".meta",".npy")])
        metadata=np.array([label,len(sentences[i].split(" "))],dtype=np.float32)
        metadata.astype(np.float32).tofile(files[cluster2file[label]])
    
def main(directory,lang,number_of_clusters,number_of_files=100):

    k=load_cluster_model(directory,"fi")
    print(k.get_params())

    # create output files
    open_files,cluster2file=create_files(directory,lang,k.get_params()["n_clusters"],number_of_files)
    
    for txt_fname in glob.glob("{}/{}_len*.txt.gz".format(directory,lang)):
        to_clusters(txt_fname,k,open_files,cluster2file)
    


   
    
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
#    g.add_argument('--length', type=str, help='Sentence length, so we know which file to read')
    g.add_argument('--dir', type=str, help='Directory to read')
    g.add_argument('--lang', type=str, help='Language (fi or en), so we know which file to read')
    g.add_argument('--files', type=int, default=10, help='How many files to use, default={n}'.format(n=10))
    g.add_argument('--clusters', type=int, default=100, help='Number of clusters, default={n}'.format(n=100))
    
    args = parser.parse_args()

    if args.dir==None or args.lang==None:
        parser.print_help()
        sys.exit(1)
    
    main(args.dir,args.lang,args.clusters,args.files)
    
    
    
    
    
