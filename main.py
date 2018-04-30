from __future__ import division

import argparse

from pychorus.helpers import find_and_output_chorus


def main(args):
    find_and_output_chorus(args.input_file, args.output_file, args.min_clip_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select and output the chorus of a piece of music")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument(
        "--output_file",
        default="chorus.wav",
        help="Output file")
    parser.add_argument(
        "--min_clip_length",
        default=15,
        help="Minimum length (in seconds) to be considered a chorus")

    main(parser.parse_args())
