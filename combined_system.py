import sys
from data_dense import iter_data
import test
import dictionary_baseline


def combined(src_fname,trg_fname,keras_model,keras_vocabulary,verbose=True):

    src_data=[]
    trg_data=[]
    for src_line,trg_line in iter_data(src_fname,trg_fname,max_pairs=1000):
        src_data.append(src_line.strip())
        trg_data.append(trg_line.strip())
    
    # keras similarities
    src_v,trg_v=test.vectorize(keras_vocabulary,src_data,trg_data,keras_model)
    all_keras_sims=test.rank(src_v,trg_v,src_data,trg_data,verbose=False) # all_keras_sims[i] is list of similarities (not sorted)
    
    # dictionary baseline similarities
    baseline_similarities=dictionary_baseline.rank(src_data,trg_data,verbose=False) # list of (idx, sim)-tuples (not sorted)
    
    ranks=[]
    for i in range(len(baseline_similarities)):
        combined_similarities=[]
        sorted_keras=[idx for idx,s in sorted([(j,sim) for j,sim in enumerate(all_keras_sims[i])], key=lambda x:x[1],reverse=True)]
        sorted_baseline=[idx for idx,s in sorted(baseline_similarities[i], key=lambda x:x[1], reverse=True)]
        for keras_sim,(j,baseline_sim) in zip(all_keras_sims[i],baseline_similarities[i]):
            combined_similarities.append((j,(keras_sim+baseline_sim)/2))
        sorted_similarities=sorted(combined_similarities, key=lambda x:x[1],reverse=True)
        if sorted_similarities[0][1]<0.4:
            continue
        result_idx=[idx for (idx,sim) in sorted_similarities]
        rank=result_idx.index(i)+1
        ranks.append(rank)
        if verbose:
            print("Source:",i,src_data[i])
            print("Reference:",trg_data[i],combined_similarities[i][1])
            print("Rank:",rank, "keras rank:",sorted_keras.index(i)+1,"baseline rank:",sorted_baseline.index(i)+1)
            for idx,score in sorted_similarities[:10]:
                print(idx,trg_data[idx],"C:",score,"K:",all_keras_sims[i][idx],"B:",baseline_similarities[i][idx][1])
            print("*"*20)
    
    print("Combined system:")    
    print("Avg:",sum(ranks)/len(ranks))
    print("#num:",len(ranks))
     
    
    
    
    
    
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    g.add_argument('-m', '--model', type=str, help='Give keras model name')
    #g.add_argument('--cutoff', type=int, default=2, help='Frequency threshold, how many times an ngram must occur to be included? (default %(default)d)')
    g.add_argument('-v', '--vocabulary', type=str, help='Give keras vocabulary file')
    
    args = parser.parse_args()

    if args.model==None or args.vocabulary==None:
        parser.print_help()
        sys.exit(1)

    combined("data/all.test.fi","data/all.test.en",args.model,args.vocabulary)

    
