import argparse

from pychorus.helpers import find_chorus




def main(args):
	find_chorus(args.input_file, args.min_clip_length)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Select and output the chorus of a piece of music")
	parser.add_argument("input_file", help="Path to input audio file")
	parser.add_argument("--min_clip_length", default=10, help="Minimum length (in seconds) to be considered a chorus")

	main(parser.parse_args())