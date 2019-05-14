import keras.models
from keras.models import Model, Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding, Average, Concatenate
from keras.layers import Dense, Input, Flatten, Dropout, Embedding, TimeDistributed, LSTM, GRU, Bidirectional
from keras import optimizers
import numpy as np
from Callbacks import Callbacks

class DnnModel():
	def __init__(self, train_embedding, embedding_size, max_sequence, paragraph_vec):
		self.model = Model()
		self.train_embedding = train_embedding
		self.embedding_size = embedding_size
		self.doc_size = 200
		self.max_sequence = max_sequence
		self.paragraph_vec = paragraph_vec
		
	def create_cnn_model(self, extra_conv, cnn_kernel, trainable):	
		if (self.paragraph_vec):
			data_dim = self.doc_size + self.embedding_size
			sequence_input = Input(shape=(self.max_sequence, data_dim,), dtype='float32')
			embedded_sequences = sequence_input
		else:
			embedding_layer = Embedding(self.train_embedding.shape[0],
                            self.embedding_size,
                            weights=[self.train_embedding],
                            input_length=self.max_sequence,
                            trainable=trainable)

			sequence_input = Input(shape=(self.max_sequence,), dtype='int32')
			embedded_sequences = embedding_layer(sequence_input)

		if (extra_conv):
			convs = []

			for filter_size in cnn_kernel:
				l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
				l_pool = MaxPooling1D(pool_size=7)(l_conv)
				convs.append(l_pool)

			l_merge = Concatenate(axis=1)(convs)
			x = Dropout(0.5)(l_merge)  
		else:
			conv = Conv1D(filters=128, kernel_size=cnn_kernel, activation='relu')(embedded_sequences)
			pool = MaxPooling1D(pool_size=3)(conv)
			x = Dropout(0.5)(pool)
			
		x = Flatten()(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.5)(x)
		preds = Dense(1, activation='sigmoid')(x)

		self.model = Model(sequence_input, preds)
		self.model.compile(loss='binary_crossentropy',
					  optimizer='adam',
					  metrics=['acc']) 	
	
	def create_rnn_model(self, rnn_type, rnn_unit, trainable):
		self.model = Sequential()
		data_dim = 0
		
		if (self.paragraph_vec):
			data_dim = self.doc_size + self.embedding_size
		else:
			data_dim = self.embedding_size
			self.model.add(Embedding(self.train_embedding.shape[0],
                            self.embedding_size,
                            weights=[self.train_embedding],
                            input_length=self.max_sequence,
                            trainable=trainable))
		
		if (rnn_type == 'bi-lstm'):
			self.model.add(Bidirectional(LSTM(rnn_unit, input_shape=(self.max_sequence, data_dim)), merge_mode='concat'))
		elif (rnn_type == 'bi-gru'):
			self.model.add(Bidirectional(GRU(rnn_unit, input_shape=(self.max_sequence, data_dim)), merge_mode='concat'))
		
		self.model.add(Dropout(0.5))
		self.model.add(Dense(1, activation='sigmoid'))
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	
	def create_hierarchy_model(self, max_sents, max_sen_len, dnn_sent_level, dnn_doc_level, trainable, **kwargs):
		
		# kwargs : cnn_kernel, lstm_unit, grnn_unit
		
		embedding_layer = Embedding(self.train_embedding.shape[0],
                            self.embedding_size,
                            weights=[self.train_embedding],
                            input_length=max_sen_len,
                            trainable=trainable)

		sentence_input = Input(shape=(max_sen_len,), dtype='int32')
		embedded_sequences = embedding_layer(sentence_input)
		sentEncoder = Model()
		
		if (dnn_sent_level == 'cnn'):
			convs = []
			filter_sizes = kwargs['cnn_kernel']

			for filter_size in filter_sizes:
				l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
				l_pool = AveragePooling1D(pool_size=3)(l_conv)
				x = Flatten()(l_pool)
				x = Dense(128, activation='tanh')(x)
				x = Dropout(0.5)(x)
				convs.append(x)

			l_average = Average()(convs)
			sentEncoder = Model(sentence_input, l_average)
		elif (dnn_sent_level == 'lstm'):
			l_lstm = LSTM(kwargs['lstm_unit'])(embedded_sequences)
			sentEncoder = Model(sentence_input, l_lstm)
		
		doc_input = Input(shape=(max_sents, max_sen_len), dtype='int32')
		doc_encoder = TimeDistributed(sentEncoder)(doc_input)
		
		if (dnn_doc_level == 'bi-lstm'):
			l_grnn_sent = Bidirectional(LSTM(kwargs['grnn_unit']))(doc_encoder)
		elif (dnn_doc_level == 'bi-gru'):
			l_grnn_sent = Bidirectional(GRU(kwargs['grnn_unit']))(doc_encoder)

		drop_1 = Dropout(0.5)(l_grnn_sent)
		preds = Dense(1, activation='sigmoid')(drop_1)
		self.model = Model(doc_input, preds)

		self.model.compile(loss='binary_crossentropy',
					  optimizer='adam',
					  metrics=['acc'])
	
	def fit(self, x_train, y_train, x_val, y_val, num_epochs, batch_size, model_path):
		callbacks = Callbacks(model_path)
		self.model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_val, y_val), 
						batch_size=batch_size, callbacks=callbacks.callbacks_list, verbose=1)
	
	def load_model(self, model_path):
		self.model = keras.models.load_model(model_path)
		
	def predict(self, x_test):
		y_pred = self.model.predict(x_test)
		return y_pred
		