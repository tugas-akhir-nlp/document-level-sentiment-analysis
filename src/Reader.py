import pandas as pd

class Reader():
	def __init__(self):
		self.file_path = ""
		self.data_frame = pd.DataFrame()
		self.hierarchy = False
		
	def sentiment_label(self, polarity):
		if polarity=='negative':
			return 0
		else:
			return 1
			
	def prepare_dataset(self):
		self.data_frame['content'] = self.data_frame['content'].astype('str')
		if(not(self.hierarchy)):
			self.data_frame['tokens'] = self.data_frame['content'].str.split()
		self.data_frame['sentiment'] = self.data_frame['polarity'].apply(self.sentiment_label)

	def read_file(self, file_path, hierarchy):
		self.file_path = file_path
		self.data_frame = pd.read_csv(self.file_path)
		self.hierarchy = hierarchy
		self.prepare_dataset()
	
	def read_corpus(self, file_path):
		self.file_path = file_path
		self.data_frame = pd.read_csv(self.file_path)
		self.data_frame['content'] = self.data_frame['content'].astype('str')