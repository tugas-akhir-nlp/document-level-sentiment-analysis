from keras.callbacks import ModelCheckpoint

class Callbacks():
	def __init__(self, model_path):
		self.checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		self.callbacks_list = [self.checkpoint]