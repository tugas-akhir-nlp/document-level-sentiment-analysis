from sklearn.cross_validation import train_test_split

class DataSplitter():
	def __init__(self):
		self.x_train = []
		self.x_validation = []
		self.y_train = [] 
		self.y_validation = []
	
	def split(self, content, polarity, test_size):
		self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(content, polarity, test_size=test_size)