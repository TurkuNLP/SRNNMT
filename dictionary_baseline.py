import sys

prob_file="lex.f2e"

finnish_vocab=set()
with open("uniq.test.tokens.100K") as f:
    for line in f:
        line=line.strip()
        finnish_vocab.add(line)

ttable={}
f=open(prob_file)
for line in f:
    f,e,p=line.strip().split()
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
    
        
#src_f="data/all.test.fi.tokenized"
#trg_f="data/all.test.en" # TODO: tokenize
src_f="data/downloads/wtower.fi-en.fi"
trg_f="data/downloads/wtower.fi-en.en" # TODO: tokenize

src_data=[]
trg_data=[]

for i,(src_sent,trg_sent) in enumerate(zip(open(src_f),open(trg_f))):
    if len(src_data)==1000:
        break
    if 5<len(src_sent.strip().split())<30 and 5<len(trg_sent.strip().split())<30:
        src_data.append(src_sent.strip())
        trg_data.append(trg_sent.strip())
    
ranks=[]
na=0
    
for i, src_sent in enumerate(src_data):
    english_words=set()
    for w in src_sent.split():
        w=w.lower()
        if w in ttable_new:
#            print("yes",w)
            english_words.update(ttable_new[w])
#        else:
#            print("no",w)
#    print(english_words)
    scores=[] 
    for j,trg_sent in enumerate(trg_data):  
        count=0
        for w in trg_sent.strip().split():
            w=w.lower()
            if w in english_words: # TODO: do set intersection
                count+=1
#        print(count)
#        print(trg_sent)
#        sys.exit()
        scores.append((j,count/len(trg_sent.split())))
    results=sorted(scores, key=lambda x:x[1], reverse=True)
    if scores[i][1]==0.0:
#        ranks.append(500)
        na+=1
        continue
        
    print("Source:",i,src_sent)
    print("Reference:",trg_data[i],scores[i][1])
    result_idx=[idx for idx,score in results]
    print("Rank:",result_idx.index(i)+1)
    ranks.append(result_idx.index(i)+1)
    for idx,score in results[:10]:
        print(idx,trg_data[idx],score)
    print("*"*20)
    
print("Avg:",sum(ranks)/len(ranks))
print("#num:",len(ranks))
print("n/a:",na)
        
#count=0    
#for key,value in ttable.items():
#    count+=len(value)
#    print(key)
#    for tr,score in sorted(value, key=lambda x: x[1],reverse=True):
#        print(tr,score)
#    print("*"*20)
##    print(key,value)
#    
#print(count)
    

