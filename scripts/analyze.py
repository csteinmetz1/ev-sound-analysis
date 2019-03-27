import os
import glob
import argparse
import numpy as np
import seaborn as sns
current_palette = sns.color_palette()
import soundfile as sf
from scipy import signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# local imports
from plot_style import *
from A_weighting import A_weighting

def analyze(audio_path, output_dir, plot_filetype):
    # does the specification say over what time period we are interested?
    block_size = 65536 # number of seconds to analyze

    #-----------------------------------------------------------------------
    # Data loading and configuration
    #-----------------------------------------------------------------------
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

    #-----------------------------------------------------------------------
    # Audio processing and analysis
    #-----------------------------------------------------------------------

    # apply A weighting filter to account for human perception
    b, a = A_weighting(fs)
    x_a = signal.lfilter(b, a, x)

    # window the signal
    w = signal.blackmanharris(n)
    x_w = x_a * w

    # perform frequency domain analysis
    y = fftpack.fft(x_w)/(n)
    y = y[range(n//2)]
    m = abs(y)

    # divide into 1/3 octave bands
    # ANSI 1/3 octave bands 11-37
    # (12.5 Hz - 5000 Hz)
    bands = np.arange(10,38)
    octaves = (2 ** ((bands-30)/3))*(1000)
    xticks = [str(int(round(x_w, 0))) for x_w in octaves][15:]

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

    # calibration
    cal = -12 # dBFS with 1 kHz sine wave @ 60dB SPL in mic location
    power += (60 - cal)

    #-----------------------------------------------------------------------
    # Plotting 
    #-----------------------------------------------------------------------

    # storage based upon input filename
    filename = os.path.basename(audio_path)
    plot_path = os.path.join(output_dir, filename.replace(".wav", ""))

    # plot joint time domain and frequency
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
    ax[0].plot(tv, x, color='#1c4b82')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude ()')
    ax[0].set_title(filename)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].bar(bands[15:], power[15:], color='#dd6b4d', zorder=3)
    ax[1].set_xticks(bands[15:])
    ax[1].set_xticklabels(xticks)
    ax[1].set_xlabel('1/3 Octave Bands - Freq (Hz)')
    ax[1].set_ylabel('Amplitude (dB SPL)')
    ax[1].axhline(y=40, color='#183661', zorder=4)
    #ax[1].grid(zorder=0)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    plt.savefig(plot_path + '_j' + '.' + plot_filetype)

    # plot frequency alone
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.bar(bands[15:], power[15:], color='#dd6b4d', zorder=3)
    ax.set_xticks(bands[15:])
    ax.set_xticklabels(xticks)
    ax.set_xlabel('1/3 Octave Bands - Freq (Hz)')
    ax.set_ylabel('Amplitude (dB SPL)')
    ax.axhline(y=40, color='#183661', zorder=4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.grid(zorder=0)
    ax.set_title(filename)

    plt.savefig(plot_path + '_f' + '.' + plot_filetype)

    # plot time domain alone
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.plot(tv, x, color='#1c4b82')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude ()')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.grid(zorder=0)
    ax.set_title(filename)

    plt.savefig(plot_path + '_t' + '.' + plot_filetype)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ev-sound-analysis")
    parser.add_argument("input", help="path to input directory (containing .wav files)", type=str)
    parser.add_argument("-o", "--output", help="path to output directory", type=str)
    parser.add_argument("-t", "--plot_filetype", help="filetype for plots ['png', 'svg', 'pdf']", type=str)

    args = parser.parse_args()

    if args.output:
        # create output directory if nonexistent 
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
    else:
        args.output = './'

    if not args.plot_filetype: # default is png
        args.plot_filetype = 'png'

    for sample in glob.glob(os.path.join(args.input, "*.wav")):
        analyze(sample, args.output, args.plot_filetype)

