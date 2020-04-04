# import python modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    '''Equation for a Gaussian
    inputs:
    H is intercept
    A is  a scaling constant
    x0 is mean
    sigma is standard deviation

    outputs: a Gaussian curve'''
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    '''Fits x and y values to a Gaussian
    inputs: x and y data
    output: the coefficients of the fitted Gaussian'''
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

def onepeak():
    '''Fits one Gaussian to data and returns FWHM'''
    filename = 'Values' # filename of file containing intensity data
    df = pd.read_csv('./{}.csv'.format(filename)) # reads data into python
    xdata = df[df.columns[0]].to_numpy() # separate the data frame into x
    ydata = df[df.columns[1]].to_numpy() # ... and y data
    H, A, x0, sigma = gauss_fit(xdata, ydata) # fits Gaussian to data and returns the variables
    FWHM = 2.355 * sigma # calculates the FWHM using Eq. 2

    # plotting the fitted gaussian with formatting
    plt.plot(xdata, ydata, 'ko', label='data')
    plt.plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), label='fit')
    plt.text(xdata[1],max(ydata),r"FWHM = {:.4f}".format(FWHM), fontsize = 12)
    plt.xlabel('Distance')
    plt.ylabel('Intensity/Gray Values')
    return FWHM

def twopeak(skew,start,end,bkg):
    '''Fits two Gaussians to data and returns FWHM
    inputs:
    skew is the number of data points away from the centre
    start is the number of data points taken at the start of the dataset
    end is the number of data points taken at the end of the dataset
    bkg is the background intensity (if known). Must be set to 0 if unknown
    output: plots a graph of two fitted Gaussians with the value of FWHM

    1. This function splits the data points in half then fits Gaussians to each half, if the data
    is skewed to one side, i.e. the midpoint of the two Gaussian peaks are not directly in the
    centre of the data points then "skew" is used. skew<0 is used if the midpoint of the peaks
    is to the left of centre and skew>0 is used fi midpoint of the peaks is right of centre.

    2. This function "zeros" the Gaussians such that the bottom of the Gaussians sit on zero.
    To find the background intensity of the data, "start" and "end" is used. These are the number
    of data points at the start and the end of the dataset used to calculate and average of the
    Gaussian tails which is the value of the background intensity. If background intensity is known
    "bkg" can be used to override the calculation based on the previous two inputs.

    3. The function then calculates the FWHM.
    '''
    filename = 'Values' # filename of file containing intensity data
    df = pd.read_csv('./{}.csv'.format(filename)) # reads data into python
    xdata = df[df.columns[0]].to_numpy() # separate the data frame into x
    ydata = df[df.columns[1]].to_numpy() # ... and y data

    # background intensity calculations
    if bkg == 0:
        if start ==0 and end == 0:
            ybackground = [0]
        elif start == 0:
            yend = ydata[-end:]
            ybackground = np.mean(yend)
        elif end == 0:
            ystart = ydata[:start]
            ybackground = np.mean(ystart)
        else:
            ystart = ydata[:start]
            yend = ydata[-end:]
            ybackground = np.mean([np.mean(ystart),np.mean(yend)])
        background = np.mean(ybackground)
    else:
        background = bkg

    ydata = ydata - background # zeros the data by subtracting background intensity
    # splitting the data points according to the location of the two Gaussians
    Npoints = len(df[df.columns[0]]) -2
    half = int(Npoints/2) + skew
    x1 = xdata[:half]
    y1 = ydata[:half]
    x2 = xdata[half:]
    y2 = ydata[half:]
    # fitting both Gaussians individually
    ygauss1 = gauss(xdata, *gauss_fit(x1, y1))
    ygauss2 = gauss(xdata, *gauss_fit(x2, y2))
    #join Gaussians together
    yg = np.concatenate((ygauss1[:half],ygauss2[half:]),axis = 0) #
    fh = np.max(yg)/2
    # calculating the FWHM
    spline2 = UnivariateSpline(xdata, yg - fh, s=0)
    R1, R2= spline2.roots() # find the roots
    FWHM = (R2-R1)
    # plotting the fitted Gaussians and the FWHM
    xfwhm = [R1,R2]
    yfwhm = [fh,fh]
    plt.plot(xfwhm,yfwhm,c = "#f55649",linewidth = 2.5)
    plt.xlabel(r"Length $\mu$m",fontsize = 12)
    plt.ylabel(r"Gray Value (Pixel Intensity)",fontsize = 12)
    plt.text(0,max(ydata),r"FWHM = {:.4f}".format(FWHM), fontsize = 12)
    vertical = [fh + 20, fh - 20]
    arm1 = [R1,R1]
    arm2 = [R2,R2]
    plt.plot(arm1, vertical,c = "#f55649",linewidth = 2.5)
    plt.plot(arm2, vertical,c = "#f55649",linewidth = 2.5)
    plt.plot(xdata, ydata, 'ko', label='data')
    plt.plot(xdata, ygauss1 , '--r')
    plt.plot(xdata, ygauss2, '--r')
    plt.xlabel('Distance')
    plt.ylabel('Intensity/gray value')
    return FWHM
