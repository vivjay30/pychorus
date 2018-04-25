import librosa
import librosa.display
import numpy as np
import scipy.signal

from math import sqrt

import matplotlib.pyplot as plt

from constants import N_FFT, SMOOTHING_SIZE_SEC


class Line(object):
	def __init__(self, start, end, lag):
		self.start = start
		self.end = end
		self.lag = lag

	def __repr__(self):
		return "Line ({} {} {})".format(self.start, self.end, self.lag)

def compute_time_lag_matrix(chroma):
	num_samples = chroma.shape[1]
	broadcast_x = np.repeat(np.expand_dims(chroma, 2) , num_samples + 1, axis=2)
	circulant_y = np.tile(chroma, (1, num_samples + 1)).reshape(12, num_samples, num_samples + 1) 
	time_lag_similarity = 1 - (np.linalg.norm((broadcast_x - circulant_y), axis=0) / sqrt(12))
	time_lag_similarity = np.rot90(time_lag_similarity, k=1, axes=(0,1))
	return time_lag_similarity[:num_samples, :num_samples]


def compute_time_time_matrix(chroma):
	broadcast_x = np.expand_dims(chroma, 2)  # (12 x n x 1)
	broadcast_y = np.swapaxes(np.expand_dims(chroma, 2), 1, 2)  # (12 x 1 x n)
	time_time_matrix = 1 - (np.linalg.norm((broadcast_x - broadcast_y), axis=0) / sqrt(12))
	return time_time_matrix

def local_maxima_rows(denoised_time_lag):
	row_sums = np.sum(denoised_time_lag, axis=1)
	divisor = np.arange(row_sums.shape[0], 0, -1)
	normalized_rows = row_sums / divisor
	local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
	return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows):
	num_samples = denoised_time_lag.shape[0]
	line_segments = []
	cur_segment_start = None
	for row in rows:
		if row < 50:
			continue
		for col in range(row, num_samples):
			if denoised_time_lag[row, col] > 0.15:
				if cur_segment_start is None:
					cur_segment_start = col
			else:
				if (cur_segment_start is not None) and (col - cur_segment_start) > 50:
					line_segments.append(Line(cur_segment_start, col, row))
					cur_segment_start = None

	return line_segments


def covering_lines(lines, margin):
	lines_dict = {}
	for line in lines:
		lines_dict[line] = 0

	# Check if line2 completely covers line 1
	for line_1 in lines:
		for line_2 in lines:
			if (line_2.start < (line_1.start + margin)) and (line_2.end > (line_1.end - margin)) and (abs(line_2.lag - line_1.lag) > 50):
				lines_dict[line_1] += 1
			if ((line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and ((line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (abs(line_2.lag - line_1.lag) > 50):
				lines_dict[line_1] += 1
	return lines_dict


def denoise_time_lag(input_matrix, time_time_matrix, smoothing_size):
	n = input_matrix.shape[0]
	horizontal_smoothing_window = np.ones((1, smoothing_size)) / smoothing_size
	horizontal_moving_average = scipy.signal.convolve2d(input_matrix, horizontal_smoothing_window, mode="full")
	left_average = horizontal_moving_average[:, 0:n]
	right_average = horizontal_moving_average[:, smoothing_size - 1:]
	max_horizontal_average = np.maximum(left_average, right_average)

	vertical_smoothing_window = np.ones((smoothing_size, 1)) / smoothing_size
	vertical_moving_average = scipy.signal.convolve2d(input_matrix, vertical_smoothing_window, mode="full")
	down_average = vertical_moving_average[0:n, :]
	up_average = vertical_moving_average[smoothing_size - 1:, :]
	

	diagonal_moving_average = scipy.signal.convolve2d(time_time_matrix, horizontal_smoothing_window, mode="full")
	ur = np.zeros((n,n))
	ll = np.zeros((n,n))
	for x in range(n):
		for y in range(x):
			ll[y,x] = diagonal_moving_average[x-y, x]
			ur[y,x] = diagonal_moving_average[x-y, x+smoothing_size - 1]

	non_horizontal_max = np.maximum.reduce([down_average, up_average, ll, ur])
	non_horizontal_min = np.minimum.reduce([up_average, down_average, ll, ur])

	suppression = (max_horizontal_average > non_horizontal_max) * non_horizontal_min +  (max_horizontal_average <= non_horizontal_max) * non_horizontal_max
	denoised_matrix = scipy.ndimage.filters.gaussian_filter1d(np.triu(input_matrix - suppression), 5*smoothing_size, axis=1)
	denoised_matrix = np.maximum(denoised_matrix, 0)
	denoised_matrix[0:5, :] = 0
	return denoised_matrix


def find_chorus(input_file, clip_length):
	print("Loading file")
	y, sr = librosa.load(input_file)
	song_length_sec = y.shape[0]/float(sr)
	S = np.abs(librosa.stft(y, n_fft=N_FFT))**2
	chroma = librosa.feature.chroma_stft(S=S, sr=sr)
	num_samples = chroma.shape[1]
	
	print("Calculating time lag similarity matrix")
	time_time_similarity = compute_time_time_matrix(chroma)
	time_lag_similarity = compute_time_lag_matrix(chroma)

	chroma_sr = num_samples/song_length_sec
	smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
	denoised_time_lag = denoise_time_lag(time_lag_similarity, time_time_similarity, smoothing_size_samples)
	rows = local_maxima_rows(denoised_time_lag)
	lines = detect_lines(denoised_time_lag, rows)

	covered_lines = covering_lines(lines, 10)
	import pdb
	pdb.set_trace()

	# librosa.display.specshow(time_lag_similarity)
	# plt.show()
	# plt.figure(figsize=(10, 4))
	librosa.display.specshow(denoised_time_lag, y_axis='time', x_axis='time', sr=2756.25) #sr=(2**14)/6)
	plt.colorbar()
	plt.set_cmap("hot_r")
	plt.show()
	# plt.colorbar()
	# plt.title('Chromagram')
	# plt.tight_layout()
	# plt.show()