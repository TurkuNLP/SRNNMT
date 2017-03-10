import sys
       
results={}     
for line in sys.stdin:
    line=line.strip()
   # try:
    sim,src,trg=line.split("\t")
    if float(sim)<0.4:
        continue
    if src not in results:
        results[src]=(sim,trg)
    elif sim>results[src][0]:
        results[src]=(sim,trg) # overwrite
    else:
        pass
   # except:
   #     break
        
        
for key,(sim,trg) in sorted(results.items(),key=lambda x:x[1][0],reverse=True):
    print(sim,key,trg,sep="\t")
