import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from math import sqrt,exp
import numpy as np
import os
SPEED_16 = 44373.9
SPEED_32 = 68594.4
BYTE = 4
os.system("clear")

def get_data(PATH,size_ar,parallel_ar,series_ar):
    file = open(PATH,"r")
    file = [line.rstrip() for line in file]
    data = []

    for line in file:
        data+=[x for x in line.split('\t')]
    for item in data:
        if item.find('size') > 0 and len(size_ar) <10 : size_ar.append(int(item[item.find('=') + 2:]))
        if item.find('series') > 0 : series_ar.append(float(item[item.find(':') + 2:]))
        elif item.find('duration') > 0 :  parallel_ar.append(float(item[item.find(':') + 2:]))
    if len(size_ar) > 0 and len(parallel_ar) > 0 and len(series_ar) > 0 : return("CORRECT")
    else : return("ERROR_WITH(get_data)")
#-----------------------------------------------------------------------------------------------#
PATH_MKL_16="/Users/konstantinPC/documents/MPI/results/clear_MKL_HALF(16)_THREAD_cache_512.txt"
PATH_MKL_32="/Users/konstantinPC/documents/MPI/results/clear_MKL_MAX(32)_THREAD_cache_512.txt"
PATH_CSR_16="/Users/konstantinPC/documents/MPI/results/OMP_CSR_HALF(16)_THREADS_cache_512.txt"
PATH_CSR_32="/Users/konstantinPC/documents/MPI/results/OMP_CSR_MAX(32)_THREADS_cache_512.txt"
PATH_NON_CSR_16="/Users/konstantinPC/documents/MPI/results/OMP_NON_CSR_HALF(16)_THREADS_cache_512.txt"
PATH_NON_CSR_32="/Users/konstantinPC/documents/MPI/results/OMP_NON_CSR_MAX(32)_THREADS_cache_512.txt"
#-----------------------------------------------------------------------------------------------#
sizes = []
parallel_duration_MKL_16 = []
series_duration_MKL_16 = []
parallel_duration_MKL_32 = []
series_duration_MKL_32 = []
parallel_duration_CSR_16 = []
series_duration_CSR_16 = []
parallel_duration_CSR_32 = []
series_duration_CSR_32 = []
parallel_duration_NON_CSR_16 = []
series_duration_NON_CSR_16 = []
parallel_duration_NON_CSR_32 = []
series_duration_NON_CSR_32 = []
print("MKL_16 : ",get_data(PATH_MKL_16,sizes,parallel_duration_MKL_16,series_duration_MKL_16))
print("MKL_32 : ",get_data(PATH_MKL_32,sizes,parallel_duration_MKL_32,series_duration_MKL_32))
print("CSR_16 : ",get_data(PATH_CSR_16,sizes,parallel_duration_CSR_16,series_duration_CSR_16))
print("CSR_32 : ",get_data(PATH_CSR_32,sizes,parallel_duration_CSR_32,series_duration_CSR_32))
print("NON_CSR_16 : ",get_data(PATH_NON_CSR_16,sizes,parallel_duration_NON_CSR_16,series_duration_NON_CSR_16))
print("NON_CSR_32 : ",get_data(PATH_NON_CSR_32,sizes,parallel_duration_NON_CSR_32,series_duration_NON_CSR_32))
#-----------------------------------------------------------------------------------------------#
non_zero_elements = [ float(5 * size_j - 4 * sqrt(size_j)) for size_j in sizes]
best_time_CSR_16=[]
best_time_NON_CSR_16=[]
best_time_CSR_32=[]
best_time_NON_CSR_32=[]
for i in range(10):
    best_time_CSR_16.append(BYTE*(2*non_zero_elements[i] + 3*sizes[i] + 1 )/ (1048576.0 * SPEED_16))
    best_time_NON_CSR_16.append(BYTE*(7*sizes[i]) / (1048576.0 * SPEED_16))
    best_time_CSR_32.append(BYTE*(2*non_zero_elements[i] + 3*sizes[i] + 1 )/ (1048576.0 * SPEED_32))
    best_time_NON_CSR_32.append(BYTE*(7*sizes[i]) / (1048576.0 * SPEED_32))

plot.title("Parallel vs In series multiplication")
#plot.legend()
line_dashed = mlines.Line2D([], [], color='black', linestyle='--', label='16 THREADS')
line = mlines.Line2D([], [], color='black', linestyle='-', label='32 THREADS')
red_patch = mpatches.Patch(color='red', label='CSR(parallel)')
grey_patch = mpatches.Patch(color='0.6', label='(series)')
green_patch = mpatches.Patch(color='green', label='MKL(parallel)')
blue_patch = mpatches.Patch(color='blue',label='NON_CSR(parallel)')
c_patch = mpatches.Patch(color='c',label='Thereotical time for CSR')
m_patch = mpatches.Patch(color='m',label='Thereotical time for NON_CSR')

plot.legend(handles=[line_dashed,line,red_patch,grey_patch,green_patch,blue_patch,c_patch,m_patch])

plot.xlabel("size")
plot.ylabel("time")
plot.yscale('log')
plot.xscale('log', basex=2)
sizes = np.array(sizes)
plot.xticks(sizes,rotation=30)
plot.plot(sizes, best_time_CSR_16,'c',linestyle='--')
plot.plot(sizes, best_time_CSR_32,'c')
plot.plot(sizes, best_time_NON_CSR_16,'m',linestyle='--')
plot.plot(sizes, best_time_NON_CSR_32,'m')
plot.plot(sizes, parallel_duration_CSR_16, 'red',linestyle='--')
plot.plot(sizes, parallel_duration_CSR_32, 'red')
plot.plot(sizes, series_duration_CSR_16, '0.6',linestyle='--')
plot.plot(sizes, series_duration_CSR_32, '0.6')
plot.plot(sizes, parallel_duration_MKL_16, 'green',linestyle='--')
plot.plot(sizes, parallel_duration_MKL_32, 'green')
plot.plot(sizes, parallel_duration_NON_CSR_16, 'blue', linestyle='--')
plot.plot(sizes, parallel_duration_NON_CSR_32, 'blue')

plot.grid(True, linestyle='-', color='0.75')
plot.show()

#exec(open("/Users/konstantinPC/documents/MPI/graphics.py").read())