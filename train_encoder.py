import logging
logging.getLogger('tensorflow').disabled = True # this removes the annoying 'Level 1:tensorflow:Registering' prints

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten, ActivityRegularization, MaxPooling1D
from keras.layers import LSTM, Bidirectional, TimeDistributed, RepeatVector


from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding

import sys
import math
import json
import numpy as np
import os


import data_dense
from models import Encoder, Decoder, EncoderDecoderModel

import tensorflow as tf
### Only needed for me, not to block the whole GPU, you don't need this stuff
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
### ---end of weird stuff



def evaluate_one(inp,model,inverse_1, inverse_2,note):
    preds=model.predict_on_batch(inp)[0]
    print("Input ({n}):".format(n=note),"".join(inverse_1([int(l) for l in inp[0] if int(l)!=0])), flush=True)
    print("Pred ({n}):".format(n=note),"".join(inverse_2([int(l) for l in np.argmax(preds, axis=-1) if l!=0])), flush=True)


def evaluate_all(vs, model, data):

    # unpack data
    (mono_src_input, mono_src_output), (mono_trg_input, mono_trg_output), (src_input, trg_input, src_output, trg_output) = data
#    inversed_source={v:k for k,v in vs.source_vocab.items()}
#    inversed_target={v:k for k,v in vs.target_vocab.items()}
    evaluate_one(mono_src_input, model.source_to_source_model, vs.inversed_vectorizer_source, vs.inversed_vectorizer_source, "fi autoencode crawl")
    evaluate_one(src_input, model.source_to_source_model, vs.inversed_vectorizer_source, vs.inversed_vectorizer_source, "fi autoencode parallel")
    evaluate_one(mono_trg_input, model.target_to_target_model, vs.inversed_vectorizer_target, vs.inversed_vectorizer_target, "en autoencode crawl")
    evaluate_one(trg_input, model.target_to_target_model, vs.inversed_vectorizer_target, vs.inversed_vectorizer_target, "en autoencode parallel")
    evaluate_one(src_input, model.source_to_target_model, vs.inversed_vectorizer_source, vs.inversed_vectorizer_target, "fi-->en")
    evaluate_one(trg_input, model.target_to_source_model, vs.inversed_vectorizer_target, vs.inversed_vectorizer_source, "en-->fi")

        

    

def train(args):
            
    #Read vocabularies
    src_f_name=args.src_train
    trg_f_name=args.trg_train
#    vs=data_dense.VocabularyChar()
#    vs=data_dense.VocabularySubWord()
    vs=data_dense.WhitespaceSeparatedVocab()
    vs.build(args.model_name+"-vocab.json", src_f_name, trg_f_name, args.monolingual_source, args. monolingual_target, force_rebuild=True) 
    vs.trainable=False

    # build model
    encoder_decoder=EncoderDecoderModel(vs.source_vocab_size, vs.target_vocab_size, args)
    encoder_decoder.build(args)

    print("Source vocabulary:", vs.source_vocab_size, "Target vocabulary:", vs.target_vocab_size)

    # data generators
    inf_iter=data_dense.infinite_iterator(src_f_name, trg_f_name, args.monolingual_source, args. monolingual_target)
    batch_iter=data_dense.fill_batch(args.minibatch_size, args.max_seq_len, vs, inf_iter)



    # save model json
#    model_json = encoder_decoder.to_json()
#    with open(args.model_name+".json", "w") as json_file:
#        json_file.write(model_json)

    # callback to save weights after each epoch
#    save_cb=ModelCheckpoint(filepath=args.model_name+".{epoch:02d}.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')

    

    counter=1
    loss={"extra_source_autoencode":[], "extra_target_autoencode":[], "parallel_source_autoencode":[], "parallel_target_autoencode":[], "parallel_source-->target":[], "parallel_target-->source":[]}
    while True:

        if counter%500==0:
            loss={"extra_source_autoencode":[], "extra_target_autoencode":[], "parallel_source_autoencode":[], "parallel_target_autoencode":[], "parallel_source-->target":[], "parallel_target-->source":[]}
            evaluate_all(vs, encoder_decoder, next(batch_iter))
            encoder_decoder.save(args.model_name)

        (mono_src_input, mono_src_output), (mono_trg_input, mono_trg_output), (src_input, trg_input, src_output, trg_output) = next(batch_iter)
        if counter%100==0:
            # train on extra monolingual data
            loss["extra_source_autoencode"].append(encoder_decoder.source_to_source_model.train_on_batch(mono_src_input, mono_src_output))
            loss["extra_target_autoencode"].append(encoder_decoder.target_to_target_model.train_on_batch(mono_trg_input, mono_trg_output))

        # monolingual parallel source
        loss["parallel_source_autoencode"].append(encoder_decoder.source_to_source_model.train_on_batch(src_input, src_output))
        # monolingual parallel target
        loss["parallel_target_autoencode"].append(encoder_decoder.target_to_target_model.train_on_batch(trg_input, trg_output))
        
        # parallel (both ways)
        loss["parallel_source-->target"].append(encoder_decoder.source_to_target_model.train_on_batch(src_input, trg_output))
        loss["parallel_target-->source"].append(encoder_decoder.target_to_source_model.train_on_batch(trg_input, src_output))
        if counter%50==0:
            avg_loss=[(key,sum(value)/len(value)) if len(value)!=0 else (key,0.0) for key, value in loss.items()]
            print("batch:", counter, "loss:", sum([v for _,v in avg_loss]),avg_loss, flush=True)
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
    g.add_argument('--max_seq_len', type=int, default=50, help='Maximum sequence length (characters)')
    g.add_argument('--embedding_size', type=int, default=250, help='Embedding size')
    g.add_argument('--recurrent_size', type=int, default=512, help='Size of the recurrent layers')
  
    
    args = parser.parse_args()

    train(args)

    # python train_encoder.py -m xxx --src_train data/europarl-v7.fi-en.fi --trg_train data/europarl-v7.fi-en.en --monolingual_source data/fi-news-crawl/all.fi.news.gz --monolingual_target data/en-news-crawl/all.en.news.gz
