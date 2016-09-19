import sys

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
       
results={}     
for (src,targets) in block_reader(sys.stdin):
    if targets[0][1]<0.4:
        continue
    
    if src not in results:
        results[src]=[]
    for (trg,sim) in targets:
        if sim<0.4:
            break
        results[src].append((trg,sim))
        
        
        
for key,values in results.items():
    for trg,sim in values:
        print(sim,key,trg,sep="||")
