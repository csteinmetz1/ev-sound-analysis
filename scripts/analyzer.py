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

# Plot size
FIGSIZE = (15, 10)

# colors
COLOR1 = "#0B3C5D"
COLOR2 = "#328CC1"
COLOR3 = "#D9B310"
COLOR4 = "#1D2731" 

# font sizes
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# plot title full names
test_names = {
    "30"   : "30km/hr",
    "20"   : "20km/hr",
    "10"   : "10km/hr",
    "stat" : "Stationary",
    "rev"  : "Reverse",
}

class Analyzer():
    
    def __init__(self, cal_file, cal_fs, cal_type, amb_file, output_dir, file_type):
        self.cal_file     = cal_file        # path to calibration nfile
        self.cal_fs       = cal_fs          # sampling rate of calibration file
        self.cal_type     = cal_type        # 'min', 'max', or 'mean'
        self.amb_file     = amb_file        # path to ambient sound level file
        self.output_dir   = output_dir      # directory to save plots
        self.file_type    = file_type       # image file type - 'png', 'jpg, 'pdf'
        #-------------------------------------------------------------------------
        self.block_size   = 65536           # analysis size
        self.cal          = self.calibrate()
        self.amb_analysis = self.analyze(self.amb_file)
        
        # load sound level requirements from json defition file
        self.slr = json.load(open("sound_level_reqs.json"))
    
    def load(self, audio_file):
        x, fs = sf.read(audio_file)
        self.validate_audio(x, fs)
        print(f"Loaded {x.shape[0]} samples with fs = {fs} from {audio_file}\n")
        return x, fs

    def calibrate(self):
        # extract the dB SPL target value measured during the calibration
        self.cal_target = int(os.path.basename(self.cal_file).split('_')[1])

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

        cal = {
            "max"  : np.max(cal_a),
            "min"  : np.min(cal_a),
            "mean" : np.mean(cal_a)}

        return cal 

    def analyze(self, audio_file):
        # get the audio samples from test file
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
        power += self.cal_target - self.cal[self.cal_type]

        analysis = {
            "audio"  : x,
            "bands"  : bands,
            "power"  : power,
            "xticks" : xticks}
        
        return analysis

    def run(self, audio_file):
        
        # determine test type based on filename
        test = self.get_test_details(audio_file)

        # get specifications based on test type
        self.band_specs = list(self.slr[test["type"]]["1/3 octave bands"].values())
        self.two_band_spec = self.slr[test["type"]]["two band spec"]

        # run analysis on test file
        analysis = self.analyze(audio_file)

        # create final plots
        plot_title     = f"{test_names[test['type']]} # {test['num']} | {test['date']}"
        plot_filename  = test['file']
        self.generate_plots(plot_title, plot_filename, analysis)

    def generate_plots(self, plot_title, plot_filename, plot_data):
        """ Create plots of the audio time series and frequency response.
        
        Params
        -------
        plot_title : string
            Title of generated plots
        plot_filename : string
            Base filename of the generated plot
        plot_data : dict
            Dictionary of data to be ploted
            
        """
        # storage based upon input filename
        plot_path = os.path.join(self.output_dir, plot_filename)

        # extract plot data
        x      = plot_data['audio']
        bands  = plot_data['bands']
        power  = plot_data['power']
        xticks = plot_data['xticks']

        # create title for each plot based on filename?

        width = 0.25

        # plotting variables
        ts = 1.0/self.fs 			    # sampling period		(seconds)
        n = x.shape[0]					# number of samples 
        t = n/self.fs					# length of signal		(seconds)
        tv = np.linspace(0, t, num=n)	# time points vector	(seconds)

        # plot joint time domain and frequency
        fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
        ax[0].plot(tv, x, color='#1c4b82')
        ax[0].set_xlim(0, t)
        ax[0].set_ylim(-1, 1)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude ()')
        ax[0].set_title(plot_title)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].grid(which='both', axis='y', linestyle='-')
        ax[0].set_yscale('symlog')

        p1 = ax[1].bar(bands[15:]-width, power[15:], width, color=COLOR1, zorder=3)
        p2 = ax[1].bar(bands[15:], self.band_specs, width,  color=COLOR2, zorder=3)
        p3 = ax[1].bar(bands[15:]+width, self.amb_analysis['power'][15:], width,  color=COLOR3, zorder=3)
        ax[1].set_xticks(bands[15:])
        ax[1].set_xticklabels(xticks)
        ax[1].set_xlabel('1/3 Octave Bands - Freq (Hz)')
        ax[1].set_ylabel('Amplitude (dB SPL)')
        p4 = ax[1].axhline(y=self.two_band_spec, color=COLOR4, zorder=4)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].legend((p1[0], p2[0], p3[0], p4), ('Measured', 'Spec', 'Ambient', '2-Band Spec'))
        ax[1].grid(which='major', axis='y', linestyle='-')
        
        plt.savefig(plot_path + '_j' + '.' + self.file_type)
        plt.cla()

        # plot frequency alone
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        p1 = ax.bar(bands[15:]-width, power[15:], width, color=COLOR1, zorder=3)
        p2 = ax.bar(bands[15:], self.band_specs, width,  color=COLOR2, zorder=3)
        p3 = ax.bar(bands[15:]+width, self.amb_analysis['power'][15:], width,  color=COLOR3, zorder=3)
        ax.set_xticks(bands[15:])
        ax.set_xticklabels(xticks)
        ax.set_xlabel('1/3 Octave Bands - Freq (Hz)')
        ax.set_ylabel('Amplitude (dB SPL)')
        p4 = ax.axhline(y=self.two_band_spec, color=COLOR4, zorder=4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend((p1[0], p2[0], p3[0], p4), ('Measured', 'Spec', 'Ambient', '2-Band Spec'))
        ax.grid(which='major', axis='y', linestyle='-')
        ax.set_title(plot_title)

        plt.savefig(plot_path + '_f' + '.' + self.file_type)
        plt.cla()

    def get_test_details(self, audio_file):
        filename = os.path.basename(audio_file).replace(".wav", "")
        details = filename.split("_")
        test = {
            "type" : details[0],
            "date" : details[1],
            "num"  : int(details[2]),
            "file" : filename}

        return test

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
