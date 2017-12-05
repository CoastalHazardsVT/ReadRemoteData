
def smooth(x,window_len=10,window='hanning'):
    #from numpy import *
    #import numpy
    from numpy import r_, convolve,ones, hanning
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
#        w=eval('numpy.'+window+'(window_len)')
        w=eval(window+'(window_len)')

#    y=numpy.convolve(w/w.sum(),s,mode='same')
    y=convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def read_data(fname):
    from numpy import loadtxt,shape,linspace
    array1 = loadtxt(fname)
    return array1


def calculate_exceedence(exce_val,scen,perc,year):
    from numpy import shape,array,sum, linspace,asarray,argwhere,zeros,concatenate

    max_data = []
    for perc_i in xrange(len(perc)):
        fname = "./data/{s1}_{per}_{ye}_b.dat".format(s1=scen,per=perc[perc_i],ye=year)
        max_data_dummy = read_data(fname)
        max_data.extend(max_data_dummy)
    max_data = asarray(max_data)
    eq = linspace(8.0,9.4,15)
    exceedenence = []
    for eq_i in xrange(len(eq)):
        exceedenence.append(len(argwhere(max_data[eq_i::15,:].reshape(5*24) >= exce_val))/(5.0*24.))
    exceedenence = asarray(exceedenence)
    exceedenence1 = zeros(len(exceedenence))
    exceedenence1[0] = exceedenence[0]
    exceedenence1[-1] = exceedenence[-1]
    for ii in xrange(10):
        exceedenence1[1:-1] = 0.5*(exceedenence[2:] + exceedenence[:-2])
    return eq, exceedenence1

def normalize(arr):
    from numpy import array,sum
    sum_arr=sum(arr)
    arr = [_/sum_arr for _ in arr]
    arr = array(arr)
    return arr

def calculate_flood_probability(scen, perc,year):
    from numpy import shape,array,sum, linspace,asarray,argwhere,zeros,concatenate
    #from pylab import *
    from numpy import histogram
    max_data = []
    for perc_i in xrange(len(perc)):
        fname = "./data/{s1}_{per}_{ye}_a.dat".format(s1=scen,per=perc[perc_i],ye=year)
        max_data_dummy = read_data(fname)
        max_data.extend(max_data_dummy)
    #    print shape(max_data)
    max_data = asarray(max_data)
    #print shape(max_data)
    max_data=max_data[:,:].reshape(5*24*15)
    #print shape(max_data)
    xx1=linspace(0,5,201)
    prob, fl = histogram(max_data, bins=xx1,normed=True)
    fl = fl[:-1] + (fl[1] - fl[0])/2
    prob_norm = normalize(prob)
    prob_smooth_norm = smooth(prob,30)
    prob_smooth_norm[prob_smooth_norm<0.0] = 0.0
    prob_smooth_norm = normalize(prob_smooth_norm)
    return fl, prob_smooth_norm
