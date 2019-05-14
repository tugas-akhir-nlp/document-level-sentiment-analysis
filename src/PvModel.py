from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
import multiprocessing
from tqdm import tqdm
from sklearn import utils
import pickle

class PvModel:
	def __init__(self):
		self.tfidf = []
		self.dbow = []
		self.dmm = []
		
	def create_pv_model(self, corpus_name, corpus):
		vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
		matrix = vectorizer.fit_transform([x.words for x in corpus])
		self.tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
	
		cores = multiprocessing.cpu_count()
		self.dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
		self.dbow.build_vocab([x for x in tqdm(corpus)])
		self.dbow.train(utils.shuffle([x for x in tqdm(corpus)]), total_examples=len(corpus), epochs=1)
		
		self.dmm = Doc2Vec(dm=1, dm_mean=1, size=100, window=4, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
		self.dmm.build_vocab([x for x in tqdm(corpus)])
		self.dmm.train(utils.shuffle([x for x in tqdm(corpus)]), total_examples=len(corpus), epochs=1)
		
		if (corpus_name == 'tripadvisor'):
			path = '../resources/vectorizer/tripadvisor/'
		elif (corpus_name == 'prosa'):
			path = '../resources/vectorizer/prosa/'
			
		with open(path + 'tfidf.pickle', 'wb') as fin:
			pickle.dump(self.tfidf, fin)
			
		self.dbow.save(path + 'model_dbow.model')
		self.dmm.save(path + 'model_dmm.model')
		
	def load_pv_model(self, corpus_name):
		if (corpus_name == 'tripadvisor'):
			path = '../resources/vectorizer/tripadvisor/'
		elif (corpus_name == 'prosa'):
			path = '../resources/vectorizer/prosa/'
			
		self.tfidf = pickle.load(open(path + 'tfidf.pickle', 'rb'))
		self.dbow = Doc2Vec.load(path + 'model_dbow.model')
		self.dmm = Doc2Vec.load(path + 'model_dmm.model')