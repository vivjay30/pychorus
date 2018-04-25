from abc import ABCMeta, abstractmethod

class SimilarityMatrix(object):
	"""Abstract class for our time-time and time-lag similarity matrices"""

	__metaclass__ = ABCMeta

	def __init__(self, chroma, sample_rate):
		self.chroma = chroma
		self.sample_rate = sample_rate # sample_rate of the audio, almost always 22050
		self.matrix = self.compute_similarity_matrix(chroma)
	
    @abstractmethod
	def compute_similarity_matrix(self, chroma):
		""""The specific type of similarity matrix we want to compute"""
		pass


	def display(self):
		import matplotlib.pyplot as plt
		librosa.display.specshow(self.matrix, y_axis='time', x_axis='time', sr=self.sample_rate/(N_FFT/2048))
		plt.colorbar()
		plt.set_cmap("hot_r")
		plt.show()


class TimeTimeSimilarityMatrix(similarity_matrix):
	"""Class for the time time similarity matrix where sample (x,y) represents how similar are the song frames x and y"""
	def compute_similarity_matrix(self, chroma):


class TimeLagSimilarityMatrix(similarity_matrix):
	"""Class to hold the time lag similarity matrix where sample (x,y) represents how similar are the song frames x and (x-y)"""

	def compute_time_lag_matrix(chroma):
		num_samples = chroma.shape[1]
		broadcast_x = np.repeat(np.expand_dims(chroma, 2) , num_samples + 1, axis=2)
		circulant_y = np.tile(chroma, (1, num_samples + 1)).reshape(12, num_samples, num_samples + 1) 
		time_lag_similarity = 1 - (np.linalg.norm((broadcast_x - circulant_y), axis=0) / sqrt(12))
		time_lag_similarity = np.rot90(time_lag_similarity, k=1, axes=(0,1))
		return time_lag_similarity[:num_samples, :num_samples]