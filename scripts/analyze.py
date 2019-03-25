import os
import glob
import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

def analyze(audio_path, output_dir):
    block_size = 65536 # number of seconds to analyze

    # load in data
    try:
        x, fs = sf.read(audio_path) # must be fs=48kHz
    except Exception as e:
        print(e)
        return

    try:
        # check for valid sample
        if x.shape[0] < block_size:
            raise RuntimeError(f"Invalid length - input file must be {block_size} samples long")
        if fs != 48000:
            raise RuntimeError("Invalid sampling rate - input file must have fs = 48kHz")
    except Exception as e:
        print(e, "\nSkipping...\n")
        return

    plot_path = os.path.join(output_dir, os.path.basename(audio_path.strip(".wav")))

    # take the left channel and block_size samples
    if x.ndim > 1:
        x = x[:block_size,0] 	
    else:
        x = x[:block_size]

    ts = 1.0/fs 					# sampling period		(seconds)
    n = x.shape[0]					# number of samples 
    t = n/fs						# length of signal		(seconds)

    tv = np.linspace(0, t, num=n)	# time points vector	(seconds)
    kv = np.arange(n)				# sample points vector 

    frq = kv/t						# frequency vector 
    frq = frq[range(n//2)]			# take just positive frequencies 

    print(f"Loaded {x.shape[0]} samples with fs = {fs} from {audio_path}\n")

    # what size fft should we use? 
    # does the specification say over what time period we are interested?

    # window the signal
    w = signal.blackmanharris(n)
    x_w = x * w

    # perform frequency domain analysis
    y = fftpack.fft(x_w)/(n)
    y = y[range(n//2)]
    m = abs(y)

    # divide into 1/3 octave bands
    # ANSI 1/3 octave bands 11-37
    # (12.5 Hz - 5000 Hz)
    bands = np.arange(10,38)
    octaves = (2 ** ((bands-30)/3))*(1000)
    xticks = [str(int(round(x_w, 0))) for x_w in octaves]

    power = []
    step = 1
    nbins = 2
    stop = 11 # starting position
    for idx, band in enumerate(bands):
        if (idx != 0) and (idx % 3 == 0):
            step *= 2
        nbins += step
        start = stop + 1
        stop  = start + nbins - 1
        with open("f.txt", "a") as f:
            f.write(f"{band} {start} {stop} {frq[start:stop+1]} {nbins}\n")
        power.append(np.sum(m[start:stop]))

    # convert amplitude to power in dB
    power = 20 * np.log10(power)

    # plot the results
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(tv,x_w)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude (dBFS)')
    ax[1].plot(bands, power, color='r')
    ax[1].set_xticks(bands)
    ax[1].set_xticklabels(xticks)
    ax[1].set_ylim(-60, 6)
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('Power (dB SPL)')

    plt.savefig(plot_path + '.png')
    plt.savefig(plot_path + '.svg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ev-sound-analysis")
    parser.add_argument("input", help="path to input directory (containing .wav files)", type=str)
    parser.add_argument("-o", "--output", help="path to output directory", type=str)
    args = parser.parse_args()

    if args.output:
        # create output directory if nonexistent 
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        args.output = './'

    for sample in glob.glob(os.path.join(args.input, "*.wav")):
        analyze(sample, args.output)

