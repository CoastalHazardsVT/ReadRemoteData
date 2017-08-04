import numpy
import os
from numpy import *
from random import randint
from matplotlib.pyplot import *
import matplotlib.patches as patches
from matplotlib import gridspec
import matplotlib.colors as colors
from pylab import *
#from mpl_toolkits.basemap import Basemap, cm
import string
import warnings
warnings.filterwarnings("ignore")
import paramiko
import string
import webbrowser
import os
from numpy import *
#from scipy.optimize import curve_fit

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.labelsize'] = 15
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['legend.fontsize'] = 12

def read_data(fname):
#    print fname
    array1 = loadtxt(fname,skiprows=1)
    return array1

def plot_diffscen(year):
    fig=plt.figure()
    fig.subplots_adjust(wspace=0.5)
    subplot(1,3,1)
    filename1 = 'LA_slr_mc_subset_rcp26_dpais.txt'
    slr_data1 = read_data(filename1)
    slr_median = []
    slr_std = []
    for j in range(20):
        slr_median.append(median(slr_data1[:,j]*1e-3))
        slr_std.append(std(slr_data1[:,j]*1e-3))
    slr_std = array(slr_std)
    slr_median = array(slr_median)
    #fill_between(years[:],slr_median-2.0*slr_std,slr_median+2.0*slr_std,facecolor='green')
    #fill_between(years[:],slr_median-slr_std,slr_median+slr_std,facecolor='blue')
    plot(years[:],slr_data1[0,:]*1e-3,'k-',lw=0.5,alpha=0.3,label=r'$\textrm{Data}$')
    plot(years[:],slr_median[:],'r-',lw=1.0,label=r'$\textrm{Median}$')
    plot(years[:],slr_median[:]+slr_std,'r--',lw=1.0,label=r'$1\sigma$')
    plot(years[:],slr_median[:]-slr_std,'r--',lw=1.0)
    plot(years[:],slr_median[:]+2.0*slr_std,'b--',lw=1.0,label=r'$2\sigma$')
    plot(years[:],slr_median[:]-2.0*slr_std,'b--',lw=1.0)

    for i in range(100):
        plot(years[:],slr_data1[i,:]*1e-3,'k-',lw=0.5,alpha=0.3)
    xlim(2000,2200)
    ylim(-0.5,5)
    #legend(loc='upper left', bbox_to_anchor=(0.0,0.99), fancybox=True, shadow=False, ncol=4, numpoints=1)
    text(2040,0.85*(5.5),r'$\textrm{RCP26}$',size=16)
    xlabel(r"$\textrm{Year}$")
    ylabel(r"$\textrm{Sea-level change [m]}$")

    subplot(1,3,2)
    filename1 = 'LA_slr_mc_subset_rcp45_dpais.txt'
    slr_data1 = read_data(filename1)
    slr_median = []
    slr_std = []
    for j in range(20):
        slr_median.append(median(slr_data1[:,j]*1e-3))
        slr_std.append(std(slr_data1[:,j]*1e-3))
    slr_std = array(slr_std)
    slr_median = array(slr_median)
    #fill_between(years[:],slr_median-2.0*slr_std,slr_median+2.0*slr_std,facecolor='green')
    #fill_between(years[:],slr_median-slr_std,slr_median+slr_std,facecolor='blue')
    plot(years[:],slr_data1[0,:]*1e-3,'k-',lw=0.5,alpha=0.3,label=r'$\textrm{Data}$')
    plot(years[:],slr_median[:],'r-',lw=1.0,label=r'$\textrm{Median}$')
    plot(years[:],slr_median[:]+slr_std,'r--',lw=1.0,label=r'$1\sigma$')
    plot(years[:],slr_median[:]-slr_std,'r--',lw=1.0)
    plot(years[:],slr_median[:]+2.0*slr_std,'b--',lw=1.0,label=r'$2\sigma$')
    plot(years[:],slr_median[:]-2.0*slr_std,'b--',lw=1.0)

    for i in range(100):
        plot(years[:],slr_data1[i,:]*1e-3,'k-',lw=0.5,alpha=0.3)
    xlim(2000,2200)
    ylim(-0.5,7.5)
    #legend(loc='upper left', bbox_to_anchor=(0.0,0.99), fancybox=True, shadow=False, ncol=4, numpoints=1)
    text(2040,0.88*(8.0),r'$\textrm{RCP45}$',size=16)
    xlabel(r"$\textrm{Year}$")
    #ylabel(r"$\textrm{Sea-level change [m]}$")


    subplot(1,3,3)
    filename1 = 'LA_slr_mc_subset_rcp85_dpais.txt'
    slr_data1 = read_data(filename1)
    slr_median = []
    slr_std = []
    for j in range(20):
        slr_median.append(median(slr_data1[:,j]*1e-3))
        slr_std.append(std(slr_data1[:,j]*1e-3))
    slr_std = array(slr_std)
    slr_median = array(slr_median)
    #fill_between(years[:],slr_median-2.0*slr_std,slr_median+2.0*slr_std,facecolor='green')
    #fill_between(years[:],slr_median-slr_std,slr_median+slr_std,facecolor='blue')
    plot(years[:],slr_data1[0,:]*1e-3,'k-',lw=0.5,alpha=0.3,label=r'$\textrm{Data}$')
    plot(years[:],slr_median[:],'r-',lw=1.0,label=r'$\textrm{Median}$')
    plot(years[:],slr_median[:]+slr_std,'r--',lw=1.0,label=r'$1\sigma$')
    plot(years[:],slr_median[:]-slr_std,'r--',lw=1.0)
    plot(years[:],slr_median[:]+2.0*slr_std,'b--',lw=1.0,label=r'$2\sigma$')
    plot(years[:],slr_median[:]-2.0*slr_std,'b--',lw=1.0)

    for i in range(100):
        plot(years[:],slr_data1[i,:]*1e-3,'k-',lw=0.5,alpha=0.3)
    xlim(2000,2200)
    ylim(-0.5,17)
    legend(loc='upper left', bbox_to_anchor=(-2.8,1.13), fancybox=True, shadow=False, ncol=4, numpoints=1)
    text(2040,0.91*(17.5),r'$\textrm{RCP85}$',size=16)
    xlabel(r"$\textrm{Year}$")
    #ylabel(r"$\textrm{Sea-level change [m]}$")
    savefig('slr_data.png',dpi=501, bbox_inches="tight")
    show()

def rcp45_sen():
    fig = figure(2,figsize = (10,5))
    fig.subplots_adjust(hspace=0.5)
    gs = gridspec.GridSpec(2,2)
    ax1= fig.add_subplot(gs[0,0])

    filename1 = 'LA_slr_mc_subset_rcp45_k2014.txt.txt'
    slr_data1 = read_data(filename1)
    slr_median = []
    slr_std = []
    for j in range(20):
        slr_median.append(median(slr_data1[:,j]*1e-3))
        slr_std.append(std(slr_data1[:,j]*1e-3))
    slr_std = array(slr_std)
    slr_median = array(slr_median)
    for i in range(100):
        ax1.plot(years[:],slr_data1[i,:]*1e-3,'k-',lw=0.5,alpha=0.3)
    #fill_between(years[:],slr_median-2.0*slr_std,slr_median+2.0*slr_std,facecolor='green')
    #fill_between(years[:],slr_median-slr_std,slr_median+slr_std,facecolor='blue')
    ax1.plot(years[:],slr_data1[0,:]*1e-3,'k-',lw=0.5,alpha=0.3,label=r'$\textrm{Data}$')
    ax1.plot(years[:],slr_median[:],'r-',lw=1.0,label=r'$\textrm{Median}$')
    ax1.plot(years[:],slr_median[:]+slr_std,'r--',lw=1.0,label=r'$1\sigma$')
    ax1.plot(years[:],slr_median[:]-slr_std,'r--',lw=1.0)
    ax1.plot(years[:],slr_median[:]+2.0*slr_std,'b--',lw=1.0,label=r'$2\sigma$')
    ax1.plot(years[:],slr_median[:]-2.0*slr_std,'b--',lw=1.0)
    slr_median1 = zeros_like(slr_median)
    slr_median1 = slr_median

    xlim(2000,2200)
    ylim(-2.5,17)
    #legend(loc='upper left', bbox_to_anchor=(-2.8,1.13), fancybox=True, shadow=False, ncol=4, numpoints=1)
    text(2070,0.71*(19.5),r'$\textrm{RCP45}$',size=16)
    xlabel(r"$\textrm{Year}$")
    ylabel(r"$\textrm{Sea-level change [m]}$")

    ax2= fig.add_subplot(gs[0,1])
    filename1 = 'LA_slr_mc_subset_rcp45_dpais.txt'
    slr_data1 = read_data(filename1)
    slr_median = []
    slr_std = []
    for j in range(20):
        slr_median.append(median(slr_data1[:,j]*1e-3))
        slr_std.append(std(slr_data1[:,j]*1e-3))
    slr_std = array(slr_std)
    slr_median = array(slr_median)
    for i in range(100):
        ax2.plot(years[:],slr_data1[i,:]*1e-3,'k-',lw=0.5,alpha=0.3)
    #fill_between(years[:],slr_median-2.0*slr_std,slr_median+2.0*slr_std,facecolor='green')
    #fill_between(years[:],slr_median-slr_std,slr_median+slr_std,facecolor='blue')
    ax2.plot(years[:],slr_data1[0,:]*1e-3,'k-',lw=0.5,alpha=0.3,label=r'$\textrm{Data}$')
    ax2.plot(years[:],slr_median[:],'r-',lw=1.0,label=r'$\textrm{Median}$')
    ax2.plot(years[:],slr_median[:]+slr_std,'r--',lw=1.0,label=r'$1\sigma$')
    plot(years[:],slr_median[:]-slr_std,'r--',lw=1.0)
    ax2.plot(years[:],slr_median[:]+2.0*slr_std,'b--',lw=1.0,label=r'$2\sigma$')
    plot(years[:],slr_median[:]-2.0*slr_std,'b--',lw=1.0)


    xlim(2000,2200)
    ylim(-2.5,17)
    legend(loc='upper left', bbox_to_anchor=(-.8,1.3), fancybox=True, shadow=False, ncol=4, numpoints=1)
    text(2050,0.71*(19.5),r'$\textrm{RCP45 with Antarc.}$',size=16)
    xlabel(r"$\textrm{Year}$")

    ax3= fig.add_subplot(gs[1,:])
    plot(years,slr_median,'b-',lw=1,label=r"$\textrm{Median: RCP45 with Antarctic Contribution}$")
    plot(years,slr_median1,'r',lw=1,label=r"$\textrm{Median: RCP45}$")
    xlabel(r"$\textrm{Year}$")
    ylabel(r"$\textrm{Sea-level change [m]}$")
    legend(loc='upper left', bbox_to_anchor=(0.001,0.98), fancybox=True, shadow=False, ncol=4, numpoints=1)
    savefig('slr_RCP45_comp.png',dpi=501, bbox_inches="tight")
    show()

def write_median(ar2,fname1):
#    print shape(ar1)
#    print shape(ar2)
    # ar3=column_stack((ar1,ar2))
    outfile=open(fname1,'w')
    savetxt(outfile, ar2,fmt='%6.3e')
    outfile.close()

def sl(x, a, b):
    return a * x + b

def expo(x, a, b,c):
    return c*(a*sqrt(2.0))**(-1.0)*exp(-0.5*((x-b)/a)**(2.0))

def expo1(x, a, b,c,d):
    return a*exp(-(x-b)**2.0 /(2.0*c**2)) + d
years = linspace(10,200,21)
years1=linspace(1,500,100)
filename1 = 'LA_slr_mc_subset_rcp45_k2014.txt.txt'
slr_data1 = read_data(filename1)
slr_median1 = []
slr_std1 = []
slr_median1.append(float(0.0))
for j in range(20):
    slr_median1.append(median(slr_data1[:,j]*1e-3))
    slr_std1.append(std(slr_data1[:,j]*1e-3))
slr_std1 = array(slr_std1)
slr_median1 = array(slr_median1)
write_median(slr_median1,'rcp45na.txt')

filename2 = 'LA_slr_mc_subset_rcp45_dpais.txt'
slr_data2 = read_data(filename2)
slr_median2 = []
slr_std2 = []
slr_median2.append(float(0.0))
for j in range(20):
    slr_median2.append(median(slr_data2[:,j]*1e-3))
    slr_std2.append(std(slr_data2[:,j]*1e-3))
slr_std2 = array(slr_std2)
slr_median2 = array(slr_median2)
write_median(slr_median2,'rcp45wa.txt')
