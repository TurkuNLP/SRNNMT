import sys
import matplotlib.pyplot as plt
import numpy as np

def tcks(ax):
    start, stop = ax.get_ylim()
    yres=(stop-start)//4
    ticks = np.arange(start, stop + yres, yres)
    ax.set_yticks(ticks)


lengths=[]
for idx,line in enumerate(sys.stdin):
    score,eng,fin=line.strip().split("\t")
    lengths.append(fin.count(" ")+1)
    if idx==1000000:
        break
f,ax=plt.subplots(4,sharex=True)
prm={"histtype":"step","bins":20,"range":(5,24),"linewidth":2,"color":"black"}
ax[0].hist(lengths[:10000],**prm)
ax[1].hist(lengths[:100000],**prm)
ax[2].hist(lengths[:500000],**prm)
ax[3].hist(lengths[:1000000],**prm)
for a,l in zip(ax,("10K","100K","500K","1000K")):
    a.set_title("Top "+l)
for a in ax:
    tcks(a)
plt.tight_layout()
plt.savefig("lenhist.pdf",format="pdf")
plt.show()


