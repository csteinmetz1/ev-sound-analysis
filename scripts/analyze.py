import numpy as np
import soundfile as sf
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

sec = 0.1 # number of seconds to analyze

# load in data
x, fs = sf.read("audio/engine.wav")
x = x[:int(fs*sec),0] 	# take the left channel and sec seconds
ts = 1.0/fs 			# sampling period		(seconds)
n = x.shape[0]			# number of samples 
t = n/fs				# length of signal		(seconds)

tv = np.linspace(0, t, num=n)	# time points vector	(seconds)
kv = np.arange(n)		# sample points vector 

frq = kv/t
frq = frq[range(n//2)]

print(f"Loaded {x.shape[0]} samples with fs = {fs}\n")

# what size fft should we use? 
# does the specification say over what time period we are intersted?

# perform frequency domain analysis
y = fftpack.fft(x)/n
y = y[range(n//2)]

# plot the results
fig, ax = plt.subplots(2, 1)
ax[0].plot(tv,x)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,np.abs(y),'r')
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()