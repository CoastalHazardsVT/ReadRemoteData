import numpy
import os
from numpy import *
from random import randint
from matplotlib.pyplot import *
#import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from pylab import *
#from mpl_toolkits.basemap import Basemap, cm
import string
import warnings
warnings.filterwarnings("ignore")
#import paramiko
import string
import webbrowser
import os
from numpy import *
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#matplotlib.rcParams['axes.labelsize'] = 18
#matplotlib.rcParams['xtick.labelsize'] = 15
#matplotlib.rcParams['ytick.labelsize'] = 15
#matplotlib.rcParams['legend.fontsize'] = 15

def smooth(x,window_len=10,window='hanning'):
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


    s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

def normalize(arr):
    sum_arr=sum(arr)
    arr = [_/sum_arr for _ in arr]
    arr = array(arr)
    return arr

def read_data_year(fname):
#    print fname
    array1 = loadtxt(fname)
    return array1.T

def read_data(fname):
#    print fname
    array1 = loadtxt(fname,skiprows=1)
    return array1


#def read_data(fname):
##    print fname
#    array1 = loadtxt(fname)
#    return array1.T

def read_SLR(fname):
    array1 = loadtxt(fname)
    return array1.T

def find_exc_prop(arr,value):
    numbereq = len(arr[:,1])
    total_real = len(arr[1,:])
    probi = zeros(numbereq)
    for i in range(numbereq):
        pp = 0.0
        for j in range(total_real):
            if arr[i,j]>=value:
                pp+=1
        probi[i] = pp / total_real
    return probi


# main code
# file_name_slr = 'rcp45na.txt'
# ar_45na = read_SLR(file_name_slr)
# print shape(ar_45na)
#
# file_name_ts = 'year_2000.dat'
# ar_2000 = read_data(file_name_ts)
#
# file_name_ts = 'year_2200.dat'
# ar_2050 = read_data(file_name_ts)
# print " 2000 \t 2050 \t Adjus \t % difference"
# print ar_45na[-1]
# for i in range(15):
#     print "{mean1:5.4f} \t {mean2:5.4f} \t {mean3:5.4f} \t {mean4:5.4f}".format(mean1=mean(ar_2000[i,:]),mean2=mean(ar_2050[i,:]), mean3=mean(ar_2050[i,:])-ar_45na[-1],mean4=100*(1.0-((mean(ar_2050[i,:])-ar_45na[-1])/mean(ar_2000[i,:]))))


year = linspace(2000,2200,21)
nyear = len(year)
neq = 15
eqf = linspace(8.0,9.4,neq)

filename2 = 'LA_slr_mc_subset_rcp45_k2014.txt.txt'
slr_data2 = read_data(filename2)

slr_data2 = slr_data2 * 1.0e-3

print shape(slr_data2)
#for i in range(100):
#	plot(slr_data2[i,:])

nreali = 24
max_val = zeros([nyear,neq,nreali])
xx=linspace(-0.0,2,31)
slr_2100, slrx_2100 = histogram(slr_data2[:,9], bins=xx,normed=True)
slrx_2100 = slrx_2100[:-1] + (slrx_2100[1] - slrx_2100[0])/2

slr_2200, slrx_2200 = histogram(slr_data2[:,19], bins=xx,normed=True)
slrx_2200 = slrx_2200[:-1] + (slrx_2200[1] - slrx_2200[0])/2

print shape(slr_2100), shape(slrx_2100)

#plot(slrx_2100,slr_2100)
print "done"

for i in range(nyear):
    #for j in range(neq):
       # dummy = '{test:3.2f}'.format(test=eqf[j])
       # mag_name = dummy.replace('.', '')
       # for k in range(nreali):
    local_f = 'year_{yeara}.dat'.format(yeara=int(year[i]))
    max_val[i,:,:] = read_data_year(local_f)
print shape(max_val)
comb_data = zeros([neq*nreali,100])
m=-1
comb_data1 = []
comb_data2 = []

for i in range(neq):
    for j in range(nreali):
        m=m+1
        for k in range(100):
            comb_data[m,k] = max_val[0,i,j]+slr_data2[k,9]
            comb_data1.append(max_val[0,i,j]+slr_data2[k,9])
            comb_data2.append(max_val[0,i,j]+slr_data2[k,19])
comb_data1 = array(comb_data1)
comb_data2 = array(comb_data2)
xx1=linspace(-0,2,31)
p_2100, x_2100 = histogram(comb_data1, bins=xx1,normed=True)
x_2100 = x_2100[:-1] + (x_2100[1] - x_2100[0])/2

p_2200, x_2200 = histogram(comb_data2, bins=xx1,normed=True)
x_2200 = x_2200[:-1] + (x_2200[1] - x_2200[0])/2


eta = p_2100 + p_2200
slr = slr_2100 + slr_2200

print shape(eta)

n_eta = normalize(eta)
print shape(p_2100),shape(x_2100)
nslr_2100 = normalize(slr_2100)
np_2100 = normalize(p_2100)
matrix = slr[:,None] * n_eta[None,:]
#plot(x_2100,p_2100)
print shape(comb_data),shape(comb_data1), shape(matrix)
figure(1)
#imshow(matrix)

cb=pcolor(xx1,xx,matrix,cmap='Blues')
cb1=colorbar(cb)
cb1.set_label('Probability')#, rotation=270)

#autoscale(False)
xlabel("Flood level")
ylabel("Sea level")
# figure(2)
# pcolor(max_val[0,:,:])
#
# figure(3)
# #for i in range(100):
# hist(slr_data2[:,9],bins=20)
# figure(4)
# plot(x_2100,p_2100)
plt.show()
#



#
#print shape(max_val)
#file_name_slr = 'rcp45wa.txt'
#ar_45na = read_SLR(file_name_slr)
#print shape(ar_45na)
#
#t_type = 2
#x1=0
#x2=15
#if t_type == 1:
#    y2000 = []
#    y2050 = []
#    y2050_a = []
#    y2100 = []
#    y2100_a = []
#    y2150 = []
#    y2150_a = []
#    y2200 = []
#    y2200_a = []
#
#
#    xy2000 = []
#    xy2050 = []
#    xy2100 = []
#    xy2150 = []
#    xy2200 = []
#    for i in range(x1,x2):
#        #for j in range(nreali):
#        y2000.append(mean(max_val[0,i,:]))
#        xy2000.append(eqf[i])
#        y2050.append(mean(max_val[5,i,:]))
#        y2050_a.append(mean(max_val[0,i,:]+ar_45na[5]))
#        xy2050.append(eqf[i])
#        y2100.append(mean(max_val[10,i,:]))
#        y2100_a.append(mean(max_val[0,i,:]+ar_45na[10]))
#        xy2100.append(eqf[i])
#        y2150.append(mean(max_val[15,i,:]))
#        y2150_a.append(mean(max_val[0,i,:]+ar_45na[15]))
#        xy2150.append(eqf[i])
#        y2200.append(mean(max_val[-1,i,:]))
#        y2200_a.append(mean(max_val[0,i,:]+ar_45na[-1]))
#        xy2200.append(eqf[i])
#
#if t_type == 2:
#    y2000 = []
#    y2050 = []
#    y2050_a = []
#    y2100 = []
#    y2100_a = []
#    y2150 = []
#    y2150_a = []
#    y2200 = []
#    y2200_a = []
#
#
#    xy2000 = []
#    xy2050 = []
#    xy2100 = []
#    xy2150 = []
#    xy2200 = []
#    for i in range(x1,x2):
#        for j in range(nreali):
#            y2000.append(max_val[0,i,j])
#            xy2000.append(eqf[i])
#            y2050.append(max_val[5,i,j])
#            y2050_a.append(max_val[0,i,j]+ar_45na[5])
#            xy2050.append(eqf[i])
#            y2100.append(max_val[10,i,j])
#            y2100_a.append(max_val[0,i,j]+ar_45na[10])
#            xy2100.append(eqf[i])
#            y2150.append(max_val[15,i,j])
#            y2150_a.append(max_val[0,i,j]+ar_45na[15])
#            xy2150.append(eqf[i])
#            y2200.append(max_val[-1,i,j])
#            y2200_a.append(max_val[0,i,j]+ar_45na[-1])
#            xy2200.append(eqf[i])
#
#
#y2000 = array(y2000)
#y2050 = array(y2050)
#y2050_ = array(y2050_a)
#y2100 = array(y2100)
#y2100_a = array(y2100_a)
#y2150 = array(y2150)
#y2150_ = array(y2150_a)
#y2200 = array(y2200)
#y2200_a = array(y2200_a)
#
#
#
#s = 1
##
#
#if s ==1:
#    y2000_s = sort(y2000)
#    y2050_s = sort(y2050)
#    y2100_s = sort(y2100)
#    y2050_a_s = sort(y2050_a)
#    y2100_a_s = sort(y2100_a)
#    y2150_s = sort(y2150)
#    y2200_s = sort(y2200)
#    y2150_a_s = sort(y2150_a)
#    y2200_a_s = sort(y2200_a)
#if s == 0:
#    y2000_s = y2000
#    y2050_s = y2050
#    y2100_s = y2100
#    y2050_a_s = y2050_a
#    y2100_a_s = y2100_a
#    y2150_s = y2150
#    y2200_s = y2200
#    y2150_a_s = y2150_a
#    y2200_a_s = y2200_a
#
##from scipy.interpolate import UnivariateSpline
#n=len(y2100_s)
#xx=linspace(0.0,4,201)
##xx=40
#p_2000, x_2000 = histogram(y2000_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2000 = x_2000[:-1] + (x_2000[1] - x_2000[0])/2   # convert bin edges to centers
#
#p_2050, x_2050 = histogram(y2050_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2050 = x_2050[:-1] + (x_2050[1] - x_2050[0])/2
#
#p_2050_a, x_2050_a = histogram(y2050_a_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2050_a = x_2050_a[:-1] + (x_2050_a[1] - x_2050_a[0])/2
#
#p_2100, x_2100 = histogram(y2100_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2100 = x_2100[:-1] + (x_2100[1] - x_2100[0])/2   # convert bin edges to centers
#
#p_2100_a, x_2100_a = histogram(y2100_a_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2100_a = x_2100_a[:-1] + (x_2100_a[1] - x_2100_a[0])/2   # convert bin edges to centers
#
#p_2150, x_2150 = histogram(y2150_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2150 = x_2150[:-1] + (x_2150[1] - x_2150[0])/2
#
#p_2150_a, x_2150_a = histogram(y2150_a_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2150_a = x_2150_a[:-1] + (x_2150_a[1] - x_2150_a[0])/2
#
#p_2200, x_2200 = histogram(y2200_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2200 = x_2200[:-1] + (x_2200[1] - x_2200[0])/2   # convert bin edges to centers
#
#p_2200_a, x_2200_a = histogram(y2200_a_s, bins=xx,normed=True) # bin it into n = N/10 bins
#x_2200_a = x_2200_a[:-1] + (x_2200_a[1] - x_2200_a[0])/2   # convert bin edges to centers
#
#
#
#
#
#np_2000 = smooth(p_2000,10)
#
#np_2050 = smooth(p_2050,10)
#np_2050[np_2050<0] =0
#np_2050_a = smooth(p_2050_a,10)
#np_2050_a[np_2050_a<0] =0
#
#
#np_2100 = smooth(p_2100,10)
#np_2100[np_2100<0] =0
#np_2100_a = smooth(p_2100_a,10)
#np_2100_a[np_2100_a<0] =0
#
#
#
#np_2150 = smooth(p_2150,10)
#np_2150[np_2150<0] =0
#np_2150_a = smooth(p_2150_a,10)
#np_2150_a[np_2150_a<0] =0
##
#
#
#np_2200 = smooth(p_2200,10)
#np_2200[np_2200<0] =0
#np_2200_a = smooth(p_2200_a,10)
#np_2200_a[np_2200_a<0] =0
#
#
#plot(x_2000,normalize(np_2000),'k-')




#plot(x_2050,normalize(np_2050),'b-')
#plot(x_2050,normalize(np_2050_a),'b:')
#
#plot(x_2100,normalize(np_2100),'r-')
#plot(x_2100,normalize(np_2100_a),'r:')


#plot(x_2150,normalize(np_2150),'g-')
#plot(x_2150,normalize(np_2150_a),'g:')


#plot(x_2200,normalize(np_2200),'y-')
#plot(x_2200,normalize(np_2200_a),'y:')



# plot(x_2050, smooth(normalize(p_2050_a)), 'b:')


# plot(x_2100, smooth(normalize(p_2100)), 'r-')
# plot(x_2100, smooth(normalize(p_2100_a)), 'r:')

# plot(x_2150, smooth(normalize(p_2150)), 'g-')
# plot(x_2150, smooth(normalize(p_2150_a)), 'g:')

# plot(x_2200, smooth(normalize(p_2200)), 'y-')
# plot(x_2200, smooth(normalize(p_2200_a)), 'y:')

#xlim(0,10)
#ylim(0,.4)
#plot(x_2000, p_2000, 'go')
#plot(n_x, g2, 'g-', linewidth=6, alpha=.6)
#plt.show()
