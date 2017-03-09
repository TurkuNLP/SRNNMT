# python 3
# read ENCOW + parsebank files, and filter it to keep only unique sentences which passes lenth and character filters
# split the senteces into size-wise batches

import sys
import math
import json
import gzip
import conllutil3 as cu
import html
import glob
import itertools
import pickle

import re


min_len=5
max_len=30

filter_regex=re.compile("[A-Za-zÅÄÖåäö]")

def good_text(sent):
    l=len(re.sub("\s+","", sent))
    if len(filter_regex.findall(sent))/l>0.8:
        return True
    else:
        return False

def read_fin_parsebank(dirname,max_sent=1000):
    total_count=0
    fnames=glob.glob(dirname+"/*.gz")
    fnames.sort()
    print(fnames,file=sys.stderr)
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr,flush=True)
        for comm, sent in cu.read_conllu(gzip.open(fname,"rt",encoding="utf-8")):
            total_count+=1
            if min_len<=len(sent)<=max_len:
                txt=" ".join(line[cu.FORM] for line in sent)
                if not good_text(txt):
                    continue
                if txt in uniq:
                    continue
                yield txt
                uniq.add(txt)
                counter+=1
            if max_sent!=0 and counter>=max_sent:
                break
    print("Fin parsebank:",total_count,file=sys.stderr)
    print("Fin parsebank yielded:",counter,file=sys.stderr)

def sent_reader(f):
    words=[]
    for line in f:
        line=line.strip()
        if line=="</s>": # end of sentence
            if words:
                yield words
                words=[]
        cols=line.split("\t")
        if len(cols)==1:
            continue
        words.append(cols[0])
        

def read_eng_parsebank(dirname,max_sent=1000):
    fnames=glob.glob(dirname+"/*.xml.gz")
    fnames.sort()
    print(fnames,file=sys.stderr)
    total_count=0
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr,flush=True)
        for sent in sent_reader(gzip.open(fname,"rt",encoding="utf-8")):
            total_count+=1
            if min_len+1<=len(sent)<=max_len:
                txt=html.unescape(" ".join(sent)) # cow corpus: &apos;
                if not good_text(txt):
                    continue
                if txt in uniq:
                    continue
                yield txt
                uniq.add(txt)
                counter+=1
            if max_sent!=0 and counter>=max_sent:
                break
    print("Eng parsebank:",total_count,file=sys.stderr)
    print("Eng parsebank yielded:",counter,file=sys.stderr)  
        

def iter_wrapper(src_dirname,trg_dirname,max_sent=10000):
    counter=0
    for fin_sent,eng_sent in itertools.zip_longest(read_fin_parsebank(src_dirname,max_sent=max_sent),read_eng_parsebank(trg_dirname,max_sent=max_sent),fillvalue="#None#"): # shorter padded with 'None'
        yield (fin_sent,eng_sent)
        counter+=1
        if max_sent!=0 and counter==max_sent:
            break
        

def filter_data(src_fname,trg_fname,max_pairs):
    # create files
    outdir="vdata_ep100k/"  
    file_dict={}
    for i in range(min_len,max_len+1): # C is here 0
        file_dict["fi_sent_len{N}".format(N=i)]=gzip.open(outdir+"/fi_len{N}.txt.gz".format(N=i,C=0),"wt",encoding="utf-8")
        file_dict["en_sent_len{N}".format(N=i)]=gzip.open(outdir+"/en_len{N}.txt.gz".format(N=i,C=0),"wt",encoding="utf-8")
              

    counter=0
    for i,(fi,en) in enumerate(iter_wrapper(src_fname,trg_fname,max_sent=max_pairs)):       
       
        if fi!="#None#":
            fi_len=len(fi.split())
            print(fi,file=file_dict["fi_sent_len{N}".format(N=fi_len)])
            
        if en!="#None#":
            en_len=len(en.split())
            print(en,file=file_dict["en_sent_len{N}".format(N=en_len)])         
            
        counter+=1
        if counter%100000==0:
            print("Processed {c} sentence pairs".format(c=counter),file=sys.stderr,flush=True)
                

    for key,value in file_dict.items():
        value.close()
     


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('--max_pairs', type=int, default=1000, help='Give max pairs of sentences to read, zero for all, default={n}'.format(n=1000))
    
    args = parser.parse_args()

    number=str(args.max_pairs) if args.max_pairs!=0 else "all"
    print("Reading",number,"sentences",file=sys.stderr)


    filter_data("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",args.max_pairs)

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




