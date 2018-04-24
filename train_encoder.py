from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten, ActivityRegularization, MaxPooling1D
from keras.layers import LSTM, Bidirectional, TimeDistributed, RepeatVector


from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.engine.topology import Layer

import sys
import math
import json
import numpy as np


import data_dense
from models import Encoder, Decoder, EncoderDecoderModel

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
### ---end of weird stuff


class CustomCallback(Callback):

    def __init__(self, dev_data,dev_labels,index2label,model_name):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

def evaluate_one(inp,model,v_input,v_pred,note):
    preds=model.predict_on_batch(inp)[0]
    print("Input ({n}):".format(n=note),"".join([v_input[int(l)] for l in inp[0] if int(l)!=0]), flush=True)
    print("Pred ({n}):".format(n=note),"".join([v_pred[l] for l in np.argmax(preds, axis=-1) if l!=0]), flush=True)


def evaluate_all(vs, model, data):

    # unpack data
    (mono_src_input, mono_src_output), (mono_trg_input, mono_trg_output), (src_input, trg_input, src_output, trg_output) = data
    inversed_source={v:k for k,v in vs.source_char.items()}
    inversed_target={v:k for k,v in vs.target_char.items()}
    evaluate_one(mono_src_input, model.source_to_source_model, inversed_source, inversed_source, "fi autoencode")
    evaluate_one(mono_trg_input, model.target_to_target_model, inversed_target, inversed_target, "en autoencode")
    evaluate_one(src_input, model.source_to_target_model, inversed_source, inversed_target, "fi-->en")
    evaluate_one(trg_input, model.target_to_source_model, inversed_target, inversed_source, "en-->fi")

        

    

def train(args):
            
    #Read vocabularies
    src_f_name=args.src_train
    trg_f_name=args.trg_train
    vs=data_dense.read_vocabularies(args.model_name+"-vocab.pickle", src_f_name, trg_f_name, args.monolingual_source, args. monolingual_target, force_rebuild=True) 
    vs.trainable=False
    print("Source characters:", len(vs.source_char), "Target characters:", len(vs.target_char))

    # build model
    encoder_decoder=EncoderDecoderModel(vs.source_char, vs.target_char, args)
    encoder_decoder.build(args)

    # data generators
    inf_iter=data_dense.InfiniteDataIterator(src_f_name, trg_f_name, args.monolingual_source, args. monolingual_target)
    batch_iter=data_dense.fill_batch(args.minibatch_size, args.max_seq_len, vs, inf_iter)



    # save model json
#    model_json = encoder_decoder.to_json()
#    with open(args.model_name+".json", "w") as json_file:
#        json_file.write(model_json)

    # callback to save weights after each epoch
#    save_cb=ModelCheckpoint(filepath=args.model_name+".{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')

    

    counter=1
    while True:

        if counter%100==0:

            evaluate_all(vs, encoder_decoder, next(batch_iter))
            encoder_decoder.save(args.model_name)

        (mono_src_input, mono_src_output), (mono_trg_input, mono_trg_output), (src_input, trg_input, src_output, trg_output) = next(batch_iter)
        loss=[]
        # monolingual source (we can run two batches...)
        loss.append(encoder_decoder.source_to_source_model.train_on_batch(mono_src_input, mono_src_output))
        loss.append(encoder_decoder.source_to_source_model.train_on_batch(src_input, src_output))
        # monolingual target
        loss.append(encoder_decoder.target_to_target_model.train_on_batch(mono_trg_input, mono_trg_output))
        loss.append(encoder_decoder.target_to_target_model.train_on_batch(trg_input, trg_output))

        # parallel (both ways)
        loss.append(encoder_decoder.source_to_target_model.train_on_batch(src_input, trg_output))
        loss.append(encoder_decoder.target_to_source_model.train_on_batch(trg_input, src_output))
        if counter%10==0:
            print("batch:", counter, "loss:", sum(loss),loss, flush=True)
        counter+=1


        # TODO: save

        # check that embeddings are shared
        #print("src-to-src:",encoder_decoder.source_to_source_model.get_layer("model_1").get_layer("char_embeddings").get_weights()[0][5][:5])
        #print("src-to-trg:",encoder_decoder.source_to_target_model.get_layer("model_1").get_layer("char_embeddings").get_weights()[0][5][:5])
        
        

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    g=parser.add_argument_group("Reguired arguments")
    
    g.add_argument('-m', '--model_name', type=str, help='Give name for trained model')
    g.add_argument('--src_train', type=str, help='Source language (Finnish) training file.')
    g.add_argument('--trg_train', type=str, help='Target language (English) training file.')
    g.add_argument('--src_devel', type=str, help='Source language (Finnish) devel file.')
    g.add_argument('--trg_devel', type=str, help='Target language (English) devel file.')
    g.add_argument('--monolingual_source', type=str, help='Monolingual data for source language (Finnish)')
    g.add_argument('--monolingual_target', type=str, help='Monolingual data for target language (English)')

    g.add_argument('--minibatch_size', type=int, default=64, help='Minibatch size')
    g.add_argument('--max_seq_len', type=int, default=100, help='Maximum sequence length (characters)')
    g.add_argument('--embedding_size', type=int, default=150, help='Embedding size')
    g.add_argument('--recurrent_size', type=int, default=512, help='Size of the recurrent layers')
  
    
    args = parser.parse_args()

    train(args)

    # python train_encoder.py -m xxx --src_train data/europarl-v7.fi-en.fi --trg_train data/europarl-v7.fi-en.en --monolingual_source data/fi-news-crawl/all.fi.news.gz --monolingual_target data/en-news-crawl/all.en.news.gz
