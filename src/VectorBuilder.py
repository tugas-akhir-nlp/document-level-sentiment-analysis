from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Doc2Vec
import pickle
import numpy as np

tfidf_trip = '../resources/vectorizer/tripadvisor/tfidf.pickle'
dbow_trip = '../resources/vectorizer/tripadvisor/model_dbow.model'
dmm_trip = '../resources/vectorizer/tripadvisor/model_dmm.model'
tfidf_prosa = '../resources/vectorizer/prosa/tfidf.pickle'
dbow_prosa = '../resources/vectorizer/prosa/model_dbow.model'
dmm_prosa = '../resources/vectorizer/prosa/model_dmm.model'

class VectorBuilder():
	def __init__(self, embedding, embedding_size, max_vocab, max_sequence, paragraph_vec, **kwargs):
		
		# kwargs : vector, corpus
		
		self.max_sequence = max_sequence
		self.tokenizer = Tokenizer()
		self.paragraph_vec = paragraph_vec
		self.doc_size = 200
		self.embedding = embedding
		self.embedding_size = embedding_size
		self.tfidf = []
		self.dbow = []
		self.dmm = []
		self.sequences = []
		self.n_data = 0
		self.texts = []
		self.lines = []
		
		if (self.paragraph_vec):
			if (kwargs['doc_vector'] == 'tripadvisor'):
				self.tfidf = pickle.load(open(tfidf_trip, 'rb'))
				self.dbow = Doc2Vec.load(dbow_trip)
				self.dmm = Doc2Vec.load(dmm_trip)
			elif (kwargs['doc_vector'] == 'prosa'):
				self.tfidf = pickle.load(open(tfidf_prosa, 'rb'))
				self.dbow = Doc2Vec.load(dbow_prosa)
				self.dmm = Doc2Vec.load(dmm_prosa)
		else:
			self.tokenizer = Tokenizer(num_words=max_vocab, lower=True, char_level=False)
			self.tokenizer.fit_on_texts(kwargs['corpus'].tolist())
	
	def build_paragraph_vector(self, tokens):
		vec = np.zeros(self.doc_size).reshape((1, self.doc_size))
		count = 0.
		for word in tokens:
			try:
				vec += np.append(self.dbow[word] * self.tfidf[word], self.dmm[word] * self.tfidf[word])
				count += 1
			except KeyError: 
				continue
		if count != 0:
			vec /= count
		return vec
	
	def build_doc_vector(self, tokens):
		doc_vec = self.build_paragraph_vector(tokens)
		data_dim = self.doc_size + self.embedding_size
		vec = np.zeros((self.max_sequence - len(tokens), data_dim))
		for word in tokens:
			try:
				word_vec = np.append(doc_vec, self.embedding[word])
				vec = np.append(vec, word_vec)
			except KeyError: 
				word_vec = np.append(doc_vec, np.zeros((1, self.embedding_size)))
				vec = np.append(vec, word_vec)
				continue
		vec.reshape(self.max_sequence, self.doc_size + self.embedding_size)
		return vec
		
	def process_token(self, tokens):
		data_dim = self.doc_size + self.embedding_size
		self.sequences[self.n_data] = self.build_doc_vector(tokens).reshape((self.max_sequence, data_dim))
		self.n_data += 1
		
	def build_sequences(self, data):
		self.sequences = []
		if (self.paragraph_vec):
			data_dim = self.doc_size + self.embedding_size
			self.sequences = np.zeros((data.shape[0], self.max_sequence, data_dim), dtype='float32')
			self.n_data = 0
			data.apply(self.process_token)
			return self.sequences
		else:			
			sequences = self.tokenizer.texts_to_sequences(data.tolist())
			self.sequences = pad_sequences(sequences, maxlen=self.max_sequence)
			return self.sequences
	
	def process_text(self, text):
		sentences = text.lower().split('.')
		self.lines.append(sentences)  
		text = text.lower().replace(".", " ")
		self.texts.append(text)
	
	def build_hierarchy_sequences(self, data, max_sents, max_sen_len):
		self.texts = []
		self.lines = []
		data.apply(self.process_text)
		word_index = self.tokenizer.word_index
		self.sequences = np.zeros((len(self.texts), max_sents, max_sen_len), dtype='int32')
		for i, sentences in enumerate(self.lines):
			for j, sent in enumerate(sentences):
				if j < max_sents:
					wordTokens = text_to_word_sequence(sent)
					k = 0
					for _, word in enumerate(wordTokens):
						if k < max_sen_len:
							self.sequences[i, j, k] = self.tokenizer.word_index[word] if word in self.tokenizer.word_index else 0
							k = k + 1
		return self.sequences