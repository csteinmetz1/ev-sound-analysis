# ev-sound-analysis
Analyzing audio from electric vehicles to determine FMVSS 141 compliance

## Overview

[idle_MITSUBISHI_PAJERO_front_engine.wav](https://freesound.org/people/gladkiy/sounds/332385/) by gladkiy was used for this example.

## Setup and Usage

Clone this repository.

```
git clone https://github.com/csteinmetz1/ev-sound-analysis
```

Install python requirements.

```
pip install -r requirements.txt
```

Run analysis on your audio file or files.

```
python scripts/evsa.py input_directory -o output_directory
```

## Background
A new regulation from the National Highway Traffic Safety Administration has been enacted that requires all electric vehicles meet given sound output requirements when operating in different scenarios. This was due to concerns that the low sound output of many electric vehicles can make them hazardous to pedestrians. The regulation provides target sound pressure levels (A weighted) over the frequency range for a number of different operating conditions of the vehicles (stationary, reverse, forward motion, etc.). Full details of the specification are presented in the document below.

[Federal Motor Vehicle Safety Standard No. 141, Minimum Sound Requirements for Hybrid and Electric Vehicles](https://www.federalregister.gov/documents/2018/02/26/2018-03721/federal-motor-vehicle-safety-standard-no-141-minimum-sound-requirements-for-hybrid-and-electric)

## Notes

* Calibration files need to be collected during every testing session
	* Play 1 kHz sine wave through speaker
	* Adjust output volume until SPL dB(A) measures 65
	* Place reference microphone in the same location as SPL meter
	* Record (at least) 5 seconds of the 1 kHz tone at same level
	* Save this calibration file with name (like cal_description_date.wav)
	* Code will load this file, find peak, and set calibration
* File naming scheme should indicate testing condition (ex: stationary, low-speed, etc.)
	* Conditions for plotting will be included 
	* Based upon test type reference lines for the requirements will change

## Resources 
* [1/3 octave band analysis by FFT](http://www.mstarlabs.com/docs/tn257.html)
* [Third-Octave Analysis Toolkit](http://www.ni.com/pdf/manuals/320952a.pdf)
* [A weighting filter](https://gist.github.com/endolith/148112) by [endolith](https://github.com/endolith)