from gensim.models.doc2vec import LabeledSentence

class DocTokenizer:
	def __init__(self):
		self.all_text = []
		
	def labelize_text(self, text):
		result = []
		prefix = 'ALL'
		for i, t in zip(text.index, text):
			result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
		return result
	
	def fit_corpus(self, corpus):
		self.all_text = self.labelize_text(corpus)
		return self.all_text