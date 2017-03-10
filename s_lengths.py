import sys
import numpy
import pickle



def gather():
    lens={} #length -> [sim,sim....]

    for line in sys.stdin:
        line=line.strip()
        cols=line.split("\t")
        if len(cols)==4: #output with numbered lines, drop
            cols=cols[1:]
        sim,en,fi=cols
        sim=float(sim)
        fi_len=fi.count(" ")+1 # num of tokens in the finnish sentence
        lens.setdefault(fi_len,[]).append(sim)

    for ln in lens:
        lens[ln]=numpy.asarray(lens[ln],numpy.float)

    mean_std={}
    for ln in lens:
        mean_std[ln]=(numpy.mean(lens[ln]),numpy.std(lens[ln]))

    for ln in sorted(mean_std):
        print(ln,mean_std[ln])
        
    with open("mean_std.pickle","wb") as f:
        pickle.dump(mean_std,f)

def normalize():
    with open("mean_std.pickle","rb") as f:
        mean_std=pickle.load(f)

    for line in sys.stdin:
        line=line.strip()
        cols=line.split("\t")
        if len(cols)==4: #output with numbered lines, drop
            cols=cols[1:]
        sim,en,fi=cols
        sim=float(sim)
        fi_len=fi.count(" ")+1 # num of tokens in the finnish sentence
        m,s=mean_std[fi_len]
        sim=(sim-m)/s
        print("{:f}".format(sim),en,fi,sep="\t")
        
    
        
if __name__=="__main__":
    #gather()
    #normalize()
    

