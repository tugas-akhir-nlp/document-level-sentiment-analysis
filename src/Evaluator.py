from sklearn.metrics import accuracy_score, classification_report

class Evaluator():
	def __init__(self, y_test):
		self.y_test = y_test
		
	def show_evaluation(self, y_pred):
		for i in range(len(y_pred)):
			y_pred[i][0] = round(y_pred[i][0])

		print("Accuracy: ", accuracy_score(self.y_test, y_pred))
		print(classification_report(self.y_test, y_pred, labels = [0, 1], digits=8))
		