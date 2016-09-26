import numpy as np
import sys
import gzip

from dictionary_baseline import build_dictionary

       

def block_reader(f):
    trg_sents=[]
    src=None
    for line in f:
        line=line.strip()
        if not line:
            yield src,trg_sents
            src=None
            trg_sents=[]
            continue      
        s,txt=line.split(" ",1)
        if s=="source:":
            src=txt
        else:
            trg_sents.append((txt,float(s)))

  
def rank_dictionary(dirname,fname):
    # fname is like 'fi_len12+en_len12'

    f2e_dictionary=build_dictionary("lex.f2e", "uniq.train.tokens.fi.100K")
    e2f_dictionary=build_dictionary("lex.e2f", "uniq.train.tokens.en.100K")
    
    src_fname,trg_fname=fname.split("+")
    src_data=[s.strip() for s in gzip.open("{D}/{F}.txt.gz".format(D=dirname,F=src_fname),"rt")]
    trg_data=[s.strip() for s in gzip.open("{D}/{F}.txt.gz".format(D=dirname,F=trg_fname),"rt")]
    
    sim_matrix=np.fromfile("{D}_out/{F}.sim.npy".format(D=dirname,F=fname),np.float32)
    sim_matrix=sim_matrix.reshape(int(len(sim_matrix)/3000),3000)
    
    idx_matrix=np.fromfile("{D}_out/{F}.idx.npy".format(D=dirname,F=fname),np.int32)
    idx_matrix=idx_matrix.reshape(int(len(idx_matrix)/3000),3000)
    
#    ranks=[]
#    na=0
#    all_scores=[]
    count=0
    print("Dictionary baseline",file=sys.stderr)
    
    for i, (index_row,sim_row) in enumerate(zip(idx_matrix,sim_matrix)):
        src_sent=src_data[i]
        english_transl=set()
        finnish_words=set(src_sent.lower().split())
        for w in finnish_words:
            if w in f2e_dictionary:
                english_transl.update(f2e_dictionary[w])

        combined=[]
        for j,(idx,s) in enumerate(zip(index_row,sim_row)):
            trg_sent=trg_data[idx]
            english_words=set(trg_sent.strip().lower().split())
            score=len(english_words&english_transl)/len(english_words) 
            finnish_transl=set()
            for w in english_words:
                if w in e2f_dictionary:
                    finnish_transl.update(e2f_dictionary[w])
            score2=len(finnish_words&finnish_transl)/len(finnish_words)
            avg=(s+score+score2)/3
            combined.append((avg,trg_sent,(s,score,score2)))
        results=sorted(combined, key=lambda x:x[0], reverse=True)
        count+=1
        if count%10000==0:
            print(count,file=sys.stderr)
        if results[0][0]<0.3: # makes no sense to keep these...
            continue

        # print
        print("source:",src_sent)
        for sim,trg,extra_sim in results[:10]:
            print(sim,extra_sim,trg,sep="\t")
        print()
        

    print("# num:",count,file=sys.stderr)


    


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-f', '--file', type=str, help='Give file name')
#    g.add_argument('-v', '--vocabulary', type=str, help='Give vocabulary file')
#    g.add_argument('--max_pairs', type=int, default=1000, help='Give vocabulary file, default={n}'.format(n=1000))
#    g.add_argument('--fi_len', type=int, help='Finnish matrix len')
#    g.add_argument('--en_len', type=int, help='English matrix len')
    
    args = parser.parse_args()

    if args.file==None:
        parser.print_help()
        sys.exit(1)



    rank_dictionary("vdata_test",args.file)
    
    

#for mx,targets in batch_iter: # input is shuffled!!!
#    src,trg=model.predict(mx)
#    print(targets,np.dot(src[0],trg[0]))
    


