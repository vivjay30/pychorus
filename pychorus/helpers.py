import librosa
import numpy as np
import scipy.signal
import soundfile as sf


from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix, Line
from pychorus.constants import N_FFT, SMOOTHING_SIZE_SEC, LINE_THRESHOLD, MIN_LINES, \
    NUM_ITERATIONS, OVERLAP_PERCENT_MARGIN


def local_maxima_rows(denoised_time_lag):
    """Find rows whose normalized sum is a local maxima"""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows, min_length_samples):
    """Detect lines in the time lag matrix. Reduce the threshold until we find enough lines"""
    cur_threshold = LINE_THRESHOLD
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95

    return line_segments


def detect_lines_helper(denoised_time_lag, rows, threshold,
                        min_length_samples):
    """Detect lines where at least min_length_samples are above threshold"""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None
                   ) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments


def count_overlapping_lines(lines, margin, min_length_samples):
    """Look at all pairs of lines and see which ones overlap vertically and diagonally"""
    line_scores = {}
    for line in lines:
        line_scores[line] = 0

    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1

    return line_scores


def best_segment(line_scores):
    """Return the best line, sorted first by chorus matches, then by duration"""
    lines_to_sort = []
    for line in line_scores:
        lines_to_sort.append((line, line_scores[line], line.end - line.start))

    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_tuple = lines_to_sort[0]
    return best_tuple[0]


def draw_lines(num_samples, sample_rate, lines):
    """Debugging function to draw detected lines in black"""
    lines_matrix = np.zeros((num_samples, num_samples))
    for line in lines:
        lines_matrix[line.lag:line.lag + 4, line.start:line.end + 1] = 1

    # Import here since this function is only for debugging
    import librosa.display
    import matplotlib.pyplot as plt
    librosa.display.specshow(
        lines_matrix,
        y_axis='time',
        x_axis='time',
        sr=sample_rate / (N_FFT / 2048))
    plt.colorbar()
    plt.set_cmap("hot_r")
    plt.show()


def create_chroma(input_file, n_fft=N_FFT):
    """
    Generate the notes present in a song

    Returns: tuple of 12 x n chroma, song wav data, sample rate (usually 22050)
             and the song length in seconds
    """
    y, sr = librosa.load(input_file)
    song_length_sec = y.shape[0] / float(sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    return chroma, y, sr, song_length_sec


def find_chorus(chroma, sr, song_length_sec, clip_length):
    """
    Find the most repeated chorus

    Args:
        chroma: 12 x n frequency chromogram
        sr: sample rate of the song, usually 22050
        song_length_sec: length in seconds of the song (lost in processing chroma)
        clip_length: minimum length in seconds we want our chorus to be (at least 10-15s)

    Returns: Time in seconds of the start of the best chorus
    """
    num_samples = chroma.shape[1]

    time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
    time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

    # Denoise the time lag matrix
    chroma_sr = num_samples / song_length_sec
    smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
    time_lag_similarity.denoise(time_time_similarity.matrix,
                                smoothing_size_samples)

    # Detect lines in the image
    clip_length_samples = clip_length * chroma_sr
    candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
    lines = detect_lines(time_lag_similarity.matrix, candidate_rows,
                         clip_length_samples)
    if len(lines) == 0:
        print("No choruses were detected. Try a smaller search duration")
        return None
    line_scores = count_overlapping_lines(
        lines, OVERLAP_PERCENT_MARGIN * clip_length_samples,
        clip_length_samples)
    best_chorus = best_segment(line_scores)
    return best_chorus.start / chroma_sr


def find_and_output_chorus(input_file, output_file, clip_length=15):
    """
    Finds the most repeated chorus from input_file and outputs to output file.

    Args:
        input_file: string specifying the input file
        output_file: string where to write the chorus (wav only)
            None means don't write anything
        clip_length: minimum length in seconds of the chorus

    Returns: Time in seconds of the start of the best chorus
    """
    chroma, song_wav_data, sr, song_length_sec = create_chroma(input_file)
    chorus_start = find_chorus(chroma, sr, song_length_sec, clip_length)
    if chorus_start is None:
        return

    print("Best chorus found at {0:g} min {1:.2f} sec".format(
        chorus_start // 60, chorus_start % 60))

    if output_file is not None:
        chorus_wave_data = song_wav_data[int(chorus_start*sr) : int((chorus_start+clip_length)*sr)]
        sf.write(output_file, chorus_wave_data, sr)
        #librosa.output.write_wav(output_file, chorus_wave_data, sr)

    return chorus_start
