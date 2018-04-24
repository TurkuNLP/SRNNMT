from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Input, MaxPooling1D
from keras.layers import Bidirectional, TimeDistributed, RepeatVector
from keras.layers import CuDNNLSTM as LSTM


from keras.callbacks import Callback,ModelCheckpoint
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.engine.topology import Layer



class Encoder(object):

    def __init__(self):
        pass

    def build(self, vocab_size, args):

        # encoder: inputs, embeddings, bilstm (return_sequences=True), lstm (return_sequences=False)
        inp=Input(shape=(args.max_seq_len,), name="character_input")
        emb=Embedding(vocab_size, args.embedding_size, name="char_embeddings")(inp)
        blstm=Bidirectional(LSTM(args.recurrent_size, return_sequences=True))(emb)
        vec=LSTM(2*args.recurrent_size, return_sequences=False)(blstm)

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
        lstm=LSTM(args.recurrent_size, return_sequences=True)(vectors)
        classification=TimeDistributed(Dense(vocab_size, activation="softmax"))(lstm)

        decoder=Model(inputs=[inp], outputs=[classification])
        print(decoder.summary())
        
        return decoder
        


class EncoderDecoderModel(object):
    """ Class to represent four different models (source-to-source, source-to-target, target-to-target, target-to-source) where encoders and decoders are shared. """

    def __init__(self, source_vocab, target_vocab, args):
        # 2 encoders (source-to-repr and target-to-repr)
        self.source_encoder=Encoder().build(len(source_vocab),args) # source language encoder
        self.target_encoder=Encoder().build(len(target_vocab),args) # target language encoder

        # 2 decoders (repr-to-source and repr-to-target)
        self.to_source_decoder=Decoder().build(len(source_vocab), args)
        self.to_target_decoder=Decoder().build(len(target_vocab), args)


    def build(self, args):

        # Full model:
        #  + two inputs (source sentence and target sentence)
        #  + two encoders
        #  + four outputs (source-to-source, source-to-target, target-to-target, target-to-source)
        #  + two decoders (representation-to-source, representation-to-target)

        # inputs
        src_input=Input(shape=(args.max_seq_len,), name="src_character_input")
        trg_input=Input(shape=(args.max_seq_len,), name="trg_character_input")

        

        # src-to-src model
        encoded=self.source_encoder(src_input)
        output=self.to_source_decoder(encoded)
        self.source_to_source_model=Model(inputs=[src_input], outputs=[output])
        self.source_to_source_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        
        # src-to-trg model
        encoded=self.source_encoder(src_input)
        output=self.to_target_decoder(encoded)
        self.source_to_target_model=Model(inputs=[src_input], outputs=[output])
        self.source_to_target_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # trg-to-trg model
        encoded=self.target_encoder(trg_input)
        output=self.to_target_decoder(encoded)
        self.target_to_target_model=Model(inputs=[trg_input], outputs=[output])
        self.target_to_target_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # trg-to-src model
        encoded=self.target_encoder(trg_input)
        output=self.to_source_decoder(encoded)
        self.target_to_source_model=Model(inputs=[trg_input], outputs=[output])
        self.target_to_source_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    def save(self, model_prefix, save_only_weights=False):
        
        print("Savin model with prefix", model_prefix, flush=True)

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

