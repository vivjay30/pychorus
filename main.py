from __future__ import division

import argparse
import requests
import os

from pychorus.helpers import find_and_output_chorus

def download_file(url, destination_folder):
    # Get the filename from the URL
    filename = url.split('/')[-1]
    # Create the destination filepath
    filepath = f'{destination_folder}/{filename}'
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(filepath, 'wb') as file:
            file.write(response.content)
        return f"File downloaded successfully as '{filename}' in '{destination_folder}'."
    else:
        return "Failed to download the file."

def main(args):
    url='https://content.spinamp.xyz/video/upload/ipfs_audio/' + args.input_cid
    download_file(url, '.')
    find_and_output_chorus(args.input_cid, None)
    if os.path.exists(args.input_cid):
        # Delete the file
        os.remove(args.input_cid)
        print("File deleted successfully.")
    else:
        print("The file does not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select and output the chorus of a piece of music")
    parser.add_argument("input_cid", help="Input CID to download and analyze")

    main(parser.parse_args())
