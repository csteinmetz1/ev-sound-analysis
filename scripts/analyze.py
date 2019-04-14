import os
import glob
import json
import argparse
import numpy as np
import seaborn as sns
import soundfile as sf
from scipy import signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# local imports
from A_weighting import A_weighting

# TODO
# ------------------------
# maybe pad with zeros if it too short
# also need to consider finding the 1 second frame with the greatest energy
# calibration loading
# plot ambient sound response as well

# Plot settings
# -----------------------------------------------------------------------------
FIGSIZE = (15, 10)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize


class Test():

    def __init__(self, audio_file):
       
        # ex: 'stat_03-30-2019_001.wav'
        self.filename = os.path.basename(audio_file).replace(".wav", "")

        # get test details from filename
        details = filename.split( "_")
        self.type = details[0]
        self.date = details[1]
        self.id  = int(details[2])

        # load audio data
        self.x, self.fs = sf.read(audio_file) # must be fs=48kHz
        self.validate_audio(self.x, self.fs)
        print(f"Loaded {self.x.shape[0]} samples with fs = {self.fs} from {audio_path}\n")

        band_specs = list(sound_level_reqs[test_type]["1/3 octave bands"].values())
        two_band_spec = sound_level_reqs[test_type]["two band spec"]

class Analyzer():
    
    def __init__(self, cal_file, cal_fs, cal_target, cal_type, block_size, output_dir, file_type):
        self.cal_file = cal_file        # path to calibratio nfile
        self.cal_fs = cal_fs            # sampling rate of calibration file
        self.cal_target = cal_target    # in dB SPL 
        self.cal_type = cal_type        # 'min', 'max', or 'mean'
        self.block_size = block_size    # analysis size
        self.output_dir = output_dir    # directory to save plots
        self.file_type = file_type      # image file type - 'png', 'jpg, 'pdf'
        self.calibrate()
        
        # load sound level requirements from json defition file
        self.slr = json.load(open("sound_level_reqs.json"))
    
    def load(self, audio_file):
        x, fs = sf.read(audio_file)
        self.validate_audio(x, fs)
        print(f"Loaded {x.shape[0]} samples with fs = {fs} from {audio_file}\n")
        return x, fs

    def calibrate(self):
        # load audio data
        self.cal_audio, self.fs = self.load(self.cal_file)

        # process calibration file
        b, a = A_weighting(self.fs)                 # create filter
        cal_a = signal.lfilter(b, a, self.cal_audio)# apply filter
        cal_a = np.power(cal_a, 2)                  # square samples
        wn = 8 / (self.fs / 2)                      # normalized frequency response
        b, a = signal.butter(2, wn)                 # create lowpass filter (125ms response time)
        cal_a = signal.lfilter(b, a, cal_a)         # apply lowpass filter
        cal_a = np.sqrt(cal_a)                      # take the sqaure root
        cal_a = 20*np.log10(cal_a)                  # convert to dB scale

        self.cal = {   
            "max"  : np.max(cal_a),
            "min"  : np.min(cal_a),
            "mean" : np.mean(cal_a)}

    def run(self, audio_file):
        
        x, fs = self.load(audio_file)

        # take the left channel and self.block_size samples
        if x.ndim > 1:
            x = x[:self.block_size,0] 	
        else:
            x = x[:self.block_size]
            
        # apply A weighting filter to account for human perception
        b, a = A_weighting(fs)
        x_a = signal.lfilter(b, a, x)

        # window the signal
        w = signal.blackmanharris(x.shape[0])
        x_w = x_a * w

        # perform frequency domain analysis
        y = fftpack.fft(x_w)/(x.shape[0])
        y = y[range(x.shape[0]//2)]
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
            power.append(np.sum(m[start:stop]))

        # convert amplitude to power in dB
        power = 20 * np.log10(power)

        # calibration - dBFS with 1 kHz sine wave @ 60dB SPL in mic location
        power += (self.cal_target - self.cal[self.cal_type])

        plot_data = {
            "audio"  : x,
            "name"   : audio_file,
            "bands"  : bands,
            "power"  : power,
            "xticks" : xticks}

        self.generate_plots("test_plot", plot_data)

    def generate_plots(self, plot_name, plot_data):

        # storage based upon input filename
        plot_path = os.path.join(self.output_dir, plot_name)

        # extract plot data
        x      = plot_data['audio']
        bands  = plot_data['bands']
        power  = plot_data['power']
        xticks = plot_data['xticks']

        # create title for each plot based on filename?

        width = 0.35

        # plotting variables
        ts = 1.0/self.fs 			    # sampling period		(seconds)
        n = x.shape[0]					# number of samples 
        t = n/self.fs					# length of signal		(seconds)
        tv = np.linspace(0, t, num=n)	# time points vector	(seconds)

        # plot joint time domain and frequency
        fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
        ax[0].plot(tv, x, color='#1c4b82')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude ()')
        ax[0].set_title(plot_name)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].bar(bands[15:], power[15:], width, color='#dd6b4d', zorder=3)
        ax[1].bar(bands[15:]+width, self.slr, width,  color='red', zorder=3)
        ax[1].set_xticks(bands[15:])
        ax[1].set_xticklabels(xticks)
        ax[1].set_xlabel('1/3 Octave Bands - Freq (Hz)')
        ax[1].set_ylabel('Amplitude (dB SPL)')
        ax[1].axhline(y=40, color='#183661', zorder=4)
        #ax[1].grid(zorder=0)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        
        plt.savefig(plot_path + '_j' + '.' + self.file_type)
        plt.cla()

        # plot frequency alone
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        p1 = ax.bar(bands[15:], power[15:], width, color='#dd6b4d', zorder=3)
        p2 = ax.bar(bands[15:]+width, self.slr, width,  color='red', zorder=3)
        ax.set_xticks(bands[15:])
        ax.set_xticklabels(xticks)
        ax.set_xlabel('1/3 Octave Bands - Freq (Hz)')
        ax.set_ylabel('Amplitude (dB SPL)')
        ax.axhline(y=40, color='#183661', zorder=4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend((p1[0], p2[0]), ('Measured', 'Spec'))
        #ax.grid(zorder=0)
        ax.set_title(plot_name)

        plt.savefig(plot_path + '_f' + '.' + self.file_type)
        plt.cla()

    def validate_audio(self, x, fs):
        """ Validate input audio data.
        
        Ensure input is numpy array of floating point data bewteen -1 and 1, 

        Params
        -------
        x : ndarray
            Input audio data
        fs : int
            Sampling rate of the input audio in Hz

        Returns
        -------
        valid : bool
            True if valid audio
            
        """
        if x.shape[0] < self.block_size:
            raise RuntimeError(f"Invalid length - input file must be {self.block_size} samples long")

        if fs != self.cal_fs:
            raise RuntimeError(f"Invalid sampling rate - input file must have fs = {self.cal_fs}")

        return True

def analyze(audio_path, output_dir, plot_filetype, sound_level_reqs):
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

    # determine test type
    # ex: 'stat_03-30-2019_001.wav'
    filename = os.path.basename(audio_path).replace(".wav", "")
    details = filename.split("_")
    test_type = details[0]
    test_date = details[1]
    test_num  = int(details[2])
    band_specs = list(sound_level_reqs[test_type]["1/3 octave bands"].values())
    two_band_spec = sound_level_reqs[test_type]["two band spec"]

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
    # create title for each plot based on filename?

    width = 0.35

    # plot joint time domain and frequency
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
    ax[0].plot(tv, x, color='#1c4b82')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude ()')
    ax[0].set_title(filename)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].bar(bands[15:], power[15:], width, color='#dd6b4d', zorder=3)
    ax[1].bar(bands[15:]+width, band_specs, width,  color='red', zorder=3)
    ax[1].set_xticks(bands[15:])
    ax[1].set_xticklabels(xticks)
    ax[1].set_xlabel('1/3 Octave Bands - Freq (Hz)')
    ax[1].set_ylabel('Amplitude (dB SPL)')
    ax[1].axhline(y=40, color='#183661', zorder=4)
    #ax[1].grid(zorder=0)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    
    plt.savefig(plot_path + '_j' + '.' + plot_filetype)
    plt.cla()

    # plot frequency alone
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    p1 = ax.bar(bands[15:], power[15:], width, color='#dd6b4d', zorder=3)
    p2 = ax.bar(bands[15:]+width, band_specs, width,  color='red', zorder=3)
    ax.set_xticks(bands[15:])
    ax.set_xticklabels(xticks)
    ax.set_xlabel('1/3 Octave Bands - Freq (Hz)')
    ax.set_ylabel('Amplitude (dB SPL)')
    ax.axhline(y=40, color='#183661', zorder=4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend((p1[0], p2[0]), ('Measured', 'Spec'))
    #ax.grid(zorder=0)
    ax.set_title(filename)

    plt.savefig(plot_path + '_f' + '.' + plot_filetype)
    plt.cla()

    # plot time domain alone
    #fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    #ax.plot(tv, x, color='#1c4b82')
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Amplitude ()')
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.grid(zorder=0)
    #ax.set_title(filename)

    #plt.savefig(plot_path + '_t' + '.' + plot_filetype)
    #plt.cla()

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
    cal_files = glob.glob(os.path.join(args.input, "cal*.wav"))
    test_files = glob.glob(os.path.join(args.input, "[!cal]*.wav"))
    if len(cal_files) < 1:
        raise RuntimeError("No calibration file found.")
    if len(test_files) < 1:
        raise RuntimeError("No test files found.")

    # create (and calibrate) analyzer
    analyzer = Analyzer(cal_files[0], 
                        48000, 
                        -60, 
                        'mean', 
                        1024, 
                        args.output, 
                        args.file_type)

    # anaylze all test files
    for sample in test_files:
        print(sample)
        analyzer.run(sample)

