# python 3
# read ENCOW + parsebank files, and filter it to keep only unique sentences which passes lenth and character filters
# split the senteces into size-wise batches

import sys
import gzip
#import conllutil3 as cu
import html
import glob
import os

import re

ID,FORM,LEMMA,UPOS,XPOS,FEAT,HEAD,DEPREL,DEPS,MISC=range(10)

min_len=5
max_len=30

filter_regex=re.compile("[A-Za-zÅÄÖåäö]")

def good_text(sent):
    l=len(re.sub("\s+","", sent))
    if len(filter_regex.findall(sent))/l>0.8:
        return True
    else:
        return False

def read_conllu(dirname,max_sent=1000):
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
                txt=" ".join(line[FORM] for line in sent)
                if not good_text(txt):
                    continue
                if txt in uniq:
                    continue
                yield txt
                uniq.add(txt)
                counter+=1
            if max_sent!=0 and counter>=max_sent:
                break
        if max_sent!=0 and counter>=max_sent:
            break
    print("conllu parsebank:",total_count,file=sys.stderr)
    print("conllu parsebank yielded:",counter,file=sys.stderr)

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
        

def read_cow(dirname,max_sent=1000):
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
            if min_len<=len(sent)<=max_len:
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
        if max_sent!=0 and counter>=max_sent:
            break
            
    print("cow parsebank:",total_count,file=sys.stderr)
    print("cow parsebank yielded:",counter,file=sys.stderr)

def read_plain(dirname,max_sent=1000):
    fnames=glob.glob(dirname+"/*.gz")
    fnames.sort()
    print(fnames,file=sys.stderr)
    total_count=0
    counter=0
    uniq=set()
    for fname in fnames:
        print(fname,file=sys.stderr,flush=True)
        for sent in gzip.open(fname,"rt",encoding="utf-8"): # one sentence per line
            sent=sent.strip()
            total_count+=1
            if min_len<=len(sent.split(" "))<=max_len:
                if not good_text(sent):
                    continue
                if sent.lower() in uniq:
                    continue
                yield sent
                uniq.add(sent.lower())
                counter+=1
            if max_sent!=0 and counter>=max_sent:
                break
        if max_sent!=0 and counter>=max_sent:
            break
            
    print("total number of sentences:",total_count,file=sys.stderr)
    print("yielded",counter,"sentences",file=sys.stderr)  
        
        
preprocessors={"conllu":read_conllu,"cow":read_cow,"plain":read_plain}
def iter_wrapper(args):

    my_preprocessor=preprocessors[args.preprocessor]
    
    for sent in my_preprocessor(args.inp,max_sent=args.max_sent):
        yield sent
        
        

def filter_data(args):
    # create files 
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    file_dict={}
    for i in range(min_len,max_len+1):
        file_dict["sent_len{N}".format(N=i)]=gzip.open(args.outdir+"/{P}_len{N}.txt.gz".format(P=args.out_prefix,N=i),"wt",encoding="utf-8")
              

    counter=0
    for i,sent in enumerate(iter_wrapper(args)):       
       
        length=len(sent.split(" ")) # simple whitespace tokenization, just used to limit comparisons 
        print(sent,file=file_dict["sent_len{N}".format(N=length)])        
            
        counter+=1
        if counter%100000==0:
            print("Processed {c} sentences".format(c=counter),file=sys.stderr,flush=True)
                

    for key,value in file_dict.items():
        value.close()
     


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('--inp', type=str, help='Input directory')
    g.add_argument('--preprocessor', type=str, help='Preprocessor to be used, options: conllu, cow or plain (one sentence per line)')
    g.add_argument('--outdir', type=str, help='Output directory name (with path)')
    g.add_argument('--out_prefix', type=str, help='Output file prefix (for example "fi" or "en")')
    g.add_argument('--max_sent', type=int, default=1000, help='Give max sentences to read, zero for all, default={n}'.format(n=1000))
    
    args = parser.parse_args()

    number=str(args.max_sent) if args.max_sent!=0 else "all"
    print("Reading",number,"sentences from",args.inp,file=sys.stderr)


    filter_data(args)

#    for s in iter_wrapper("/home/jmnybl/git_checkout/SRNNMT/parsebank","/home/jmnybl/git_checkout/SRNNMT/EN-COW",max_sent=10000000):
#        pass




