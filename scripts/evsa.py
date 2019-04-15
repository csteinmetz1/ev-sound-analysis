import os
import glob
import argparse
from analyzer import Analyzer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ev-sound-analysis")
    parser.add_argument("input", help="path to input directory (containing .wav files)", type=str)
    parser.add_argument("-o", "--output", help="path to output directory", type=str)
    parser.add_argument("-t", "--file_type", help="filetype for plots ['png', 'svg', 'pdf']", type=str)
    args = parser.parse_args()

    if args.output:
        # create output directory if nonexistent 
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        args.output = './'

    if not args.file_type: # default is png
        args.file_type = 'png'

    # get audio files 
    cal_files  = glob.glob(os.path.join(args.input, "cal*.wav"))
    amb_files  = glob.glob(os.path.join(args.input, "amb*.wav"))
    test_files = glob.glob(os.path.join(args.input, "[!cal]*.wav"))

    if len(cal_files) < 1:
        raise RuntimeError("No calibration file found.")
    if len(amb_files) < 1:
        raise RuntimeError("No ambient test files found.")
    if len(test_files) < 1:
        raise RuntimeError("No test files found.")

    # create (and calibrate) analyzer
    analyzer = Analyzer(cal_files[0], 48000, 'mean', # calibration file details
                        amb_files[0],                # ambient file details   
                        args.output, args.file_type) # output plot setttings

    # anaylze all test files
    for sample in test_files:
        try:
            print(sample)
            analyzer.run(sample)
        except Exception as e:
            print(e)