import sys
from data_dense import iter_data


def build_dictionary(translation_file,uniq_tokens):

    vocabulary=set()
    with open(uniq_tokens) as f:
        for line in f:
            line=line.strip().lower()
            vocabulary.add(line)

    ttable={}
    f=open(translation_file)
    for line in f:
        f,e,p=line.strip().lower().split()
        if f not in vocabulary:
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
        
    return ttable_new # key: word, value: list of translations
    


def rank(src_data,trg_data,verbose=True):

#    f2e_dictionary=build_dictionary("bible_data/bible.lex.f2e", "bible_data/bible.uniq.train.tokens.fi.100K")
#    e2f_dictionary=build_dictionary("bible_data/bible.lex.e2f", "bible_data/bible.uniq.train.tokens.en.100K")

    f2e_dictionary=build_dictionary("europarl100K_exp/ep100k.lex.f2e", "europarl100K_exp/uniq.train.tokens.fi.100K")
    e2f_dictionary=build_dictionary("europarl100K_exp/ep100k.lex.e2f", "europarl100K_exp/uniq.train.tokens.en.100K")
    
#    f2e_dictionary=build_dictionary("europarl500K_exp/europarl500K_exp/e500k.lex.f2e", "europarl500K_exp/europarl500K_exp/uniq.train.tokens.fi.100K")
#    e2f_dictionary=build_dictionary("europarl500K_exp/europarl500K_exp/e500k.lex.e2f", "europarl500K_exp/europarl500K_exp/uniq.train.tokens.en.100K")
    
#    f2e_dictionary=build_dictionary("lex.f2e", "uniq.train.tokens.fi.100K")
#    e2f_dictionary=build_dictionary("lex.e2f", "uniq.train.tokens.en.100K")
    
    ranks=[]
    na=0
    all_scores=[]   
    
    for i, src_sent in enumerate(src_data):
        english_transl=set()
        finnish_words=set(src_sent.lower().split())
        for w in finnish_words:
            if w in f2e_dictionary:
                english_transl.update(f2e_dictionary[w])

        scores=[]
        scores2=[]
        for j,trg_sent in enumerate(trg_data):  
            count=0
            english_words=set(trg_sent.strip().lower().split())
            score=len(english_words&english_transl)
    #        print(score,len(english_words),score/len(english_words))
          #  continue
          #  print(len(english_words))
          #  continue 
            scores.append((j,score/len(english_words)))
            finnish_transl=set()
            for w in english_words:
                if w in e2f_dictionary:
                    finnish_transl.update(e2f_dictionary[w])
            score2=len(finnish_words&finnish_transl)
            scores2.append((j,score2/len(finnish_words)))
        #break
        combined=[(x,(f+e)/2,(f,e)) for (x,f),(y,e) in zip(scores,scores2)]
        results=sorted(combined, key=lambda x:x[1], reverse=True)
        all_scores.append(combined)
        if combined[i][1]==0.0: # TODO
            ranks.append(len(src_data)/2)
            na+=1
            continue
        result_idx=[idx for idx,score,sec_score in results]
        ranks.append(result_idx.index(i)+1)
        if verbose:
            print("Source:",i,src_sent)
            print("Reference:",trg_data[i],combined[i][1],combined[i][2])
            print("Rank:",result_idx.index(i)+1)
            for idx,score,sec_score in results[:10]:
                print(idx,trg_data[idx],score,sec_score)
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
    for src_line,trg_line in iter_data(src_fname,trg_fname,max_pairs=5000):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())

    
    all_similarities=rank(src_data,trg_data)
     
     
     
if __name__=="__main__":

#    test("europarl100K_exp/all.test.new.fi.tokenized","data/all.test.new.en.tokenized")
#    test("europarl100K_exp/ep100k.en-fi.fi.tok.devel","europarl100K_exp/ep100k.en-fi.en.tok.devel")
    test("data/wmttest.fi-en.fi.tokenized","data/wmttest.fi-en.en.tokenized")        

    

