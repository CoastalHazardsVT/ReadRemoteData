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
    array1 = loadtxt(fname,skiprows=2)
    return max(array1[:,5])

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

#nfiles = 10
#ndir=6
# year = ['1788','1813','1838']
#year = ['1838']
# year = linspace(2000,2200,21)
# nyear = len(year)
# #eq = ['80','82','84','86','88','90','92','94']
# neq = 15
# eqf = linspace(8.0,9.4,neq)
#
#
# nreali = 24
# max_val = zeros([nyear,neq,nreali])
# # machine_n = 'dragonstooth1.arc.vt.edu'
# # ssh = paramiko.SSHClient()
# # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# # ssh.connect(machine_n, username='weiszr', password='uJQWNk.VJ_)js]a2HH')
# # print "Downloading data..."
# # ftp = ssh.open_sftp()
# for i in range(nyear):
#     for j in range(neq):
#         dummy = '{test:3.2f}'.format(test=eqf[j])
#         mag_name = dummy.replace('.', '')
#         for k in range(nreali):
#             # if year[i] == 1788:
#             #     seal = 0.0
#             # if year[i] == 2000:
#             #     seal = 0.3
#             # if year[i] == 2100:
#             #     seal = 1.08
#             path = str('/work/dragonstooth/weiszr/SLR/run_RCP45NA_{yeara}_{eqa}_{test1:04d}/_output/gauge10010.txt'.format(yeara=int(year[i]),eqa=mag_name,test1=k))
#             local_f = 'gauge1_RCP45NA_{yeara}_{eqa}_{test1:04d}.dat'.format(yeara=int(year[i]),eqa=mag_name,test1=k)
#             print "\t",path," -> ", local_f
#             # ftp.get(path,local_f)
#             max_val[i,j,k] = read_data(local_f)
# # ?\ftp.close
# fig = figure(1,figsize = (25,5))
# #fig.tight_layout()
# for j in range(15):
#     subplot(1,15,j+1)
#     for i in range(nreali):
#         plot(year[:],max_val[:,j,i],'ko')
#         ylim(0.0,2)
#     # for i in range(nreali):
#     #     plot(year[0],max_val[0,j,i],'ko')
#     #     ylim(0.0,2)
#     # for i in range(nreali):
#     #     plot(year[2],max_val[2,j,i],'ko')
#     #     ylim(0.0,2)
#     title('{test:3.2f}'.format(test=eqf[j]))
# savefig('data.png',dpi=501, bbox_inches="tight")
# # exc_1788_032 = find_exc_prop(max_val[0,:,:],0.30)
# # exc_2000_032 = find_exc_prop(max_val[1,:,:],0.30)
# # exc_2100_032 = find_exc_prop(max_val[2,:,:],0.30)
# # figure(2)
# # plot(eqf,exc_1788_032,'k-',label = '1788')
# # plot(eqf,exc_2000_032,'b-',label = '1813')
# # plot(eqf,exc_2100_032,'r-',label = '1838')
# # xlabel('Magnitude')
# # ylabel('Exceedance Probability (0.32m)')
# # legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
# #        ncol=3, borderaxespad=0.)
# # savefig('exccedance_032.png',dpi=501, bbox_inches="tight")
# #
# # exc_1788_110 = find_exc_prop(max_val[0,:,:],0.1)
# # exc_2000_110 = find_exc_prop(max_val[1,:,:],0.1)
# # exc_2100_110 = find_exc_prop(max_val[2,:,:],0.1)
# # figure(3)
# # plot(eqf,exc_1788_110,'k-',label = '1788')
# # plot(eqf,exc_2000_110,'b-',label = '1813')
# # plot(eqf,exc_2100_110,'r-',label = '1828')
# # xlabel('Magnitude')
# # ylabel('Exceedance Probability (1.1m)')
# # legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
# #        ncol=3, borderaxespad=0.)
# # savefig('exccedance_110.png',dpi=501, bbox_inches="tight")
# #
#
#
# # subplot(8,1,2)
# # plot(max_val[1,1,:],'ko')
# # subplot(8,1,3)
# # plot(max_val[1,2,:],'ko')
# # subplot(8,1,4)
# # plot(max_val[1,3,:],'ko')
# # subplot(8,1,5)
# # plot(max_val[1,4,:],'ko')
# # subplot(8,1,6)
# # plot(max_val[1,5,:],'ko')
# # subplot(8,1,7)
# # plot(max_val[1,6,:],'ko')
# # subplot(8,1,8)
# # plot(max_val[1,7,:],'ko')
# # show()
# print "..Done!"
