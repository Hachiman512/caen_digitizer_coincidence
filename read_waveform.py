import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal
from numpy.typing import NDArray

def rough_sinc_interp(samples, freq_s_ratio):
    offset_amount = int(len(samples)/2)
    padded_samples = np.concatenate([ offset_amount*[samples[0]], samples, offset_amount*[samples[-1]]])
    f_s = int(freq_s_ratio * len(padded_samples))
    resamples = scipy.signal.resample(padded_samples, f_s)
    print(len(samples))
    T_s = 1/f_s
    t = np.arange(0, 1, T_s)
    y = np.zeros(len(t))
    for k in range(1, len(resamples)):
        y = y + resamples[k] * np.sinc((t - k*T_s)/T_s)
    return scipy.signal.resample(y, len(padded_samples))[offset_amount:-offset_amount]

def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    assert len(x) == len(s), 'x and s must be the same length'
    
    # Find the period    
    T = s[1] - s[0]
    sincM = np.tile(u,(len(s),1))- np.tile(s[:,np.newaxis],(1,len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y

def sinc_interpolation(x: NDArray, s: NDArray, u: NDArray) -> NDArray:
            """Whittaker–Shannon or sinc or bandlimited interpolation.
            Args:
                x (NDArray): signal to be interpolated, can be 1D or 2D
                s (NDArray): time points of x (*s* for *samples*) 
                u (NDArray): time points of y (*u* for *upsampled*)
            Returns:
                NDArray: interpolated signal at time points *u*
            Reference:
                This code is based on https://gist.github.com/endolith/1297227
                and the comments therein.
            TODO:
                * implement FFT based interpolation for speed up
            """
            print(type(None))
            sinc_ = np.sinc((u - s[:,None])/(s[1]-s[0]))

            return np.dot(x, sinc_)

os.chdir('/home/anil/software/coincidence/test_junk')
cwd = os.getcwd()
print("Current working directory is : {0}".format(cwd))

f_data= open("channel0.CSV")
data = np.genfromtxt(f_data, delimiter=';')
f_data.close()
print(np.shape(data))
trace=data[2,6:150]
time=np.zeros(len(trace))
tstamp=0.0
count=(np.shape(trace))
print(len(trace))

for i in range(len(trace)):
    time[i]=tstamp
    tstamp=tstamp+4.0E-09

print(type(time))
stime = np.arange(time[0],time[-1],2.0E-09)
fun = sinc_interpolation(trace, time,stime)
#print(len(rough_sinc_interp(temp, freq_s_ratio=0.5)))
plt.plot(time,trace,"b+",markersize=6)
plt.plot(stime, sinc_interpolation(trace,time,stime),"ro",markersize=2)
plt.show()
