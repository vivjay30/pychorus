# Pychorus

Pychorus is an open source library to find choruses or interesting sections in pieces of music. The algorithm is largely based on [a paper](https://pdfs.semanticscholar.org/f120/3fb2efe2f251ea7c221c9eaca95cc163594b.pdf) by Masataka Goto with some simplifications and modifications. There is room for improvement so feel free to contribute to the project.

Check out my blog post: https://towardsdatascience.com/finding-choruses-in-songs-with-python-a925165f94a8 for a full explanation on how the library works

## Getting Started

You can install the codebase easily with

```
pip install pychorus
```

### Sample execution

The most straightforward way to use the module is as follows:

```
from pychorus import find_and_output_chorus

chorus_start_sec = find_and_output_chorus("path/to/audio_file", "path/to/output_file", clip_length)
```

You can also clone the repo and use main.py as a command line tool like
```
python main.py path/to/audio_file --output_file=path/to/output_file
```

### Creating the chromogram, time-time, and time-lag matrices

```
from pychorus import create_chroma
from pychorus.similarity_matrix import TimeTimeSimilarityMatrix, TimeLagSimilarityMatrix

chroma, _, sr, _ = create_chroma("path/to/audio_file")
time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

# Visualize the results
time_time_similarity.display()
time_lag_similarity.display()
```

## Planned improvements for v0.2
* Detect choruses in music recorded without a metronome by looking for slightly crooked lines
* API to return all choruses, not just one with the most matches
* Add ability to output entire detected chorus, not just section of size clip_length

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
