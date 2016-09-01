import sys
from data_dense import iter_data


def build_dictionary():
    prob_file="lex.f2e"

    finnish_vocab=set()
    with open("uniq.test.tokens.100K") as f:
        for line in f:
            line=line.strip().lower()
            finnish_vocab.add(line)

    ttable={}
    f=open(prob_file)
    for line in f:
        f,e,p=line.strip().lower().split()
        if f not in finnish_vocab:
            continue
        if float(p)<0.001 or float(p)==1.0:
            continue
        if f in ttable:
            ttable[f].append((e,float(p)))
        else:
            ttable[f]=[(e,float(p))]

    ttable_new={}        
    for key,value in ttable.items():
        values=sorted(value, key=lambda x:x[1], reverse=True)
        val=[v for v,s in values[:30]]
        ttable_new[key]=set(val)
        
    return ttable_new # key: Finnish word, value: list of English translations
    


def rank(src_data,trg_data,verbose=True):

    dictionary=build_dictionary()
    
    ranks=[]
    na=0
    all_scores=[]   
    
    for i, src_sent in enumerate(src_data):
        english_words=set()
        for w in src_sent.lower().split():
            if w in dictionary:
                english_words.update(dictionary[w])

        scores=[] 
        for j,trg_sent in enumerate(trg_data):  
            count=0
            words=set(trg_sent.strip().lower().split())
            score=len(words&english_words) 

            scores.append((j,score/len(words)))
        results=sorted(scores, key=lambda x:x[1], reverse=True)
        all_scores.append(scores)
        if scores[i][1]==0.0: # TODO
            ranks.append(len(src_data)/2)
            na+=1
            continue
        result_idx=[idx for idx,score in results]
        ranks.append(result_idx.index(i)+1)
        if verbose:
            print("Source:",i,src_sent)
            print("Reference:",trg_data[i],scores[i][1])
            print("Rank:",result_idx.index(i)+1)
            for idx,score in results[:10]:
                print(idx,trg_data[idx],score)
            print("*"*20)
    print("Dictionary baseline:")    
    print("Avg:",sum(ranks)/len(ranks))
    print("#num:",len(ranks))
    print("n/a:",na)
    
    return all_scores
    
    
def test(src_fname,trg_fname):

    # read sentences
    src_data=[]
    trg_data=[]
    for src_line,trg_line in iter_data(src_fname,trg_fname,max_pairs=1000):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())

    
    all_similarities=rank(src_data,trg_data)
     
     
     
if __name__=="__main__":

    test("data/all.test.fi","data/all.test.en")        

    

