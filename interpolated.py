'''
Description: Tsinc interpolation
Author: Ming Fang
Date: 2022-04-20 17:07:21
LastEditors: Ming Fang
LastEditTime: 2022-04-20 17:22:38
'''
import numpy as np
from pathlib import Path
from numba import njit
import matplotlib.pylab as plt


@njit
def _tsincInterpolate(p, numInterp, tsincWidth, tsincCoefs):
    """A helper function with njit acceleration to perform tsinc interpolation.

    Args:
        p (np.ndarray): Input signal
        numInterp (int): number of time intervals between two samples in input signal
        tsincWidth (int): Width of tsinc interpolation function.
        tsincCoefs (np.ndarray): Discretized tsinc function, to be convoluted with input signal

    Returns:
        np.ndarray: Interpolated function.
    """
    interPulseLen = (len(p)-1) * numInterp
    interpolatedPulse = np.zeros(interPulseLen)
    for i in range(interPulseLen):
        k = i % numInterp
        j = i // numInterp
        if k == 0:
            interpolatedPulse[i] = p[j]
        else:
            tmp = 0
            for l in range(tsincWidth):
                if j+1+l < len(p):
                    tmp += p[j+1+l] * tsincCoefs[(l+1)*numInterp - k]
                if j - l >= 0:
                    tmp += p[j-l] * tsincCoefs[l * numInterp + k]
            interpolatedPulse[i] = tmp
    return interpolatedPulse


class TSincInterpolator:
    """Interpolate pulse using a terminated sinc function.
    """
    def __init__(self, numInterP:int, tsincWidth:int=6, taperConst:int=30):
        """Initialize the pulse interpolator.

        Args:
            numInterP (int): number of parts a time step is evenly divided into after interpolation
            tsincWidth (int): number of lobes of tsinc function used in interpolation, default to 6
            taperConst (int): tapering constant, default=30
        """
        self.numInterP = numInterP
        self.tsincWidth = tsincWidth
        self.taperConst = taperConst
        self.sincCoefs = np.zeros(tsincWidth * numInterP)
        self._getTSincCoefficients()

    def _getTSincCoefficients(self):
        """Calculate the tsinc function values.
        """
        self.sincCoefs[0] = 1
        for j in range(1, self.tsincWidth * self.numInterP):
            phi = j * np.pi / self.numInterP
            tmp = j / self.taperConst
            tmp = np.sin(phi) / phi * np.exp(-tmp**2)
            self.sincCoefs[j] = tmp

    def getInterpolatedPulse(self, p):
        """Interpolate the input signal using tsinc interpolation.

        Args:
            p (np.ndarray): Input signal.

        Returns:
            np.ndarray: Interpolated signal, with length = numInterp * (len(p)-1)
        """
        return _tsincInterpolate(p, self.numInterP, self.tsincWidth, self.sincCoefs)


if __name__ == "__main__":
    numInterp = 8
    tSincInterpolator = TSincInterpolator(numInterp)
    # load input pulse
    timeStep = 4  # ns
    inputPulse = np.loadtxt("baf2.txt")
    # interpolation
    interpolatedPulse = tSincInterpolator.getInterpolatedPulse(inputPulse)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(timeStep * np.arange(len(inputPulse)), inputPulse, label="Input")
    # ax.plot(timeStep / numInterp * np.arange(len(interpolatedPulse)), interpolatedPulse, label="Interpolated")
    ax.scatter(timeStep / numInterp * np.arange(len(interpolatedPulse)), interpolatedPulse, s=5, label="Interpolated")
    ax.scatter(timeStep * np.arange(len(inputPulse)), inputPulse, marker='s', s=5, label="Input")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    plt.show()
