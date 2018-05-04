from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, GlobalMaxPooling1D
from keras.layers import Bidirectional, TimeDistributed, RepeatVector
from keras.layers import CuDNNLSTM as LSTM
from keras.optimizers import Adam
from keras import regularizers

from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.engine.topology import Layer



class Encoder(object):

    def __init__(self):
        pass

    def build(self, vocab_size, args):

        # encoder: inputs, embeddings, bilstm (return_sequences=True), lstm (return_sequences=False)
        inp=Input(shape=(args.max_seq_len,), name="input")
        emb=Embedding(vocab_size, args.embedding_size, name="embeddings")(inp)
        drop=Dropout(0.2)(emb)
        blstm=Bidirectional(LSTM(args.recurrent_size, return_sequences=True, activity_regularizer=regularizers.l1(10e-5)))(drop)
        
        vec=GlobalMaxPooling1D()(blstm)
        #vec=LSTM(2*args.recurrent_size, return_sequences=False)(blstm)

        # ...encoder ready
        encoder=Model(inputs=[inp], outputs=[vec])
        print(encoder.summary())
        
        return encoder

class Decoder(object):

    def __init__(self):
        pass

    def build(self, vocab_size, args):

        # decoder: RepeatVector, LSTM, Timedistributed, classification
        inp=Input((2*args.recurrent_size,))
        vectors=RepeatVector(args.max_seq_len)(inp)
        lstm=LSTM(2*args.recurrent_size, return_sequences=True)(vectors)
        classification=TimeDistributed(Dense(vocab_size, activation="softmax"))(lstm)

        decoder=Model(inputs=[inp], outputs=[classification])
        print(decoder.summary())
        
        return decoder
        


class EncoderDecoderModel(object):
    """ Class to represent four different models (source-to-source, source-to-target, target-to-target, target-to-source) where encoders and decoders are shared. """

    def __init__(self, source_vocab_size, target_vocab_size, args):
        # 2 encoders (source-to-repr and target-to-repr)
        self.source_encoder=Encoder().build(source_vocab_size,args) # source language encoder
        self.target_encoder=Encoder().build(target_vocab_size,args) # target language encoder

        # 2 decoders (repr-to-source and repr-to-target)
        self.to_source_decoder=Decoder().build(source_vocab_size, args)
        self.to_target_decoder=Decoder().build(target_vocab_size, args)


    def build(self, args):

        # Full model:
        #  + two inputs (source sentence and target sentence)
        #  + two encoders
        #  + four outputs (source-to-source, source-to-target, target-to-target, target-to-source)
        #  + two decoders (representation-to-source, representation-to-target)

        # inputs
        src_input=Input(shape=(args.max_seq_len,), name="src_input")
        trg_input=Input(shape=(args.max_seq_len,), name="trg_input")

        

        # src-to-src model
        encoded=self.source_encoder(src_input)
        output=self.to_source_decoder(encoded)
        self.source_to_source_model=Model(inputs=[src_input], outputs=[output])
        optim=Adam(lr=args.learning_rate,amsgrad=True)
        self.source_to_source_model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, sample_weight_mode='temporal')
        
        # src-to-trg model
        encoded=self.source_encoder(src_input)
        output=self.to_target_decoder(encoded)
        self.source_to_target_model=Model(inputs=[src_input], outputs=[output])
        optim=Adam(lr=args.learning_rate,amsgrad=True)
        self.source_to_target_model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, sample_weight_mode='temporal')

        # trg-to-trg model
        encoded=self.target_encoder(trg_input)
        output=self.to_target_decoder(encoded)
        self.target_to_target_model=Model(inputs=[trg_input], outputs=[output])
        optim=Adam(lr=args.learning_rate,amsgrad=True)
        self.target_to_target_model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, sample_weight_mode='temporal')

        # trg-to-src model
        encoded=self.target_encoder(trg_input)
        output=self.to_source_decoder(encoded)
        self.target_to_source_model=Model(inputs=[trg_input], outputs=[output])
        optim=Adam(lr=args.learning_rate,amsgrad=True)
        self.target_to_source_model.compile(loss="sparse_categorical_crossentropy", optimizer=optim, sample_weight_mode='temporal')

    def save(self, model_prefix, save_only_weights=False):
        
        print("Saving model with prefix", model_prefix, flush=True)

        # save encoders and decoders (all weights are there...)
        if not save_only_weights:
            # save model json
            model_json = self.source_encoder.to_json()
            with open(model_prefix+"-source_encoder.json", "w") as json_file:
                json_file.write(model_json)
            model_json = self.target_encoder.to_json()
            with open(model_prefix+"-target_encoder.json", "w") as json_file:
                json_file.write(model_json)
            model_json = self.to_source_decoder.to_json()
            with open(model_prefix+"-to_source_decoder.json", "w") as json_file:
                json_file.write(model_json)
            model_json = self.to_target_decoder.to_json()
            with open(model_prefix+"-to_target_decoder.json", "w") as json_file:
                json_file.write(model_json)
        # save weights
        self.source_encoder.save_weights(model_prefix+"-source_encoder.h5")
        self.target_encoder.save_weights(model_prefix+"-target_encoder.h5")
        self.to_source_decoder.save_weights(model_prefix+"-to_source_decoder.h5")
        self.to_target_decoder.save_weights(model_prefix+"-to_target_decoder.h5")

