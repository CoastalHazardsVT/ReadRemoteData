import numpy
import os
from numpy import *
from random import randint
from matplotlib.pyplot import *
import matplotlib.patches as patches
import matplotlib.colors as colors
from pylab import *
from mpl_toolkits.basemap import Basemap, cm
import string
import warnings
warnings.filterwarnings("ignore")
import paramiko
import string
import webbrowser
import os
from numpy import *
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15

def read_data(fname):
#    print fname
    array1 = loadtxt(fname)
    return array1.T

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
file_name_slr = 'rcp45na.txt'
ar_45na = read_SLR(file_name_slr)
print shape(ar_45na)

file_name_ts = 'year_2000.dat'
ar_2000 = read_data(file_name_ts)

file_name_ts = 'year_2050.dat'
ar_2050 = read_data(file_name_ts)
print " 2000 \t 2050 \t Adjus \t % difference"
#print ar_45na[5]
for i in range(15):
    print "{mean1:5.4f} \t {mean2:5.4f} \t {mean3:5.4f} \t {mean4:5.4f}".format(mean1=mean(ar_2000[i,:]),mean2=mean(ar_2050[i,:]), mean3=mean(ar_2050[i,:])-ar_45na[5],mean4=100*(1.0-((mean(ar_2050[i,:])-ar_45na[5])/mean(ar_2000[i,:]))))
