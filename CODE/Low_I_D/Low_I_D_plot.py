"""
Synthetic test 3

A Python program to generate the input for the
Synthetic test 3 – Simulated field at low latitude and remanent magnetization

This code is released from the paper: 
Amplitude of the magnetic anomaly vector in the interpretation
 of total-field anomaly data at low magnetic latitudes

The program is under the conditions terms in the file README.txt

authors: Felipe F. Melo and Shayane P. Gonzalez, 2019
email: felipe146@hotmail.com, shayanegonzalez@gmail.com 


Input:
    - input_mag.dat
        The input data generated from generate_input.py

    - parameters.dat 
    	The parameters estimated with the equivalent layer.

    - predict.dat 
    	The predicted data, from the total field-anomaly, with the equivalent
        layer.

    - amplitude.dat 
    	The amplitude of the magnetic anomaly vector. 

Output:
    Figures: 
        Fig_4.png (a) Total-field anomaly. (b) Total-gradient. 
        (c) Amplitude of the magnetic anomaly vector. 
        
        Fig_4b.png (a) Predicted data. (b) Residuals. 
        (c) Histogram of the residuals.    
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from fatiando.gravmag import transform
import matplotlib.patches as patches
import gc
import matplotlib.mlab as mlab

# Create a regular grid at -150 m height
shape = (116, 97)

input_data=np.loadtxt('input_mag.dat')

amplitude_input=np.loadtxt('amplitude.dat')

predict_input=np.loadtxt('predict.dat')

parameters_input=np.loadtxt('parameters.dat')

yi=input_data[:,0]
xi=input_data[:,1]
tf=input_data[:,2]
amplitude=amplitude_input[:,2]
predict=predict_input[:,2]
parameters=parameters_input[:,2]

'''
We need the geometry for define the polygons on the plots
'''
# Dipping source
vety=np.linspace(5000,9000,41)
y1m=np.zeros((vety.size-1))
y2m=np.zeros((vety.size-1))
for i,c in enumerate(vety):
    if i == 40:
        pass
    else:
        y1m[i]=vety[i]
        y2m[i]=vety[i+1]

#top 
zo_t=np.linspace(100,2000,40)  
#base
zo_b=np.linspace(200,2100,40)  
# south, north
x1, x2=5000,9000

#L shape source
x1_L, x2_L = 26000, 30000
y1_L, y2_L = 8000, 8500
#Top and bottom of 1st the prism
z1_L, z2_L = 200, 1100
x3_L, x4_L = 29500, 30000
y3_L, y4_L = 8500, 10500
#Top and bottom of 2nd the prism
z3_L, z4_L = 200, 1100

#three sources
# 1st prism
x1_c, x2_c = 20000, 24000
y1_c, y2_c = 16000, 24000
#Top and bottom of 1st the prism
z1_c, z2_c = 1000, 5000
# 2nd prism
x3_c, x4_c = 21500, 22500
y3_c, y4_c = 21500, 22500
#Top and bottom of 2nd the prism
z3_c, z4_c = 200, 1000
# 3rd prism 
y5_c, y6_c = 17500, 18500

#vertical prism
x1_cr, x2_cr = 11000, 13000
y1_cr, y2_cr = 19000, 21000
z1_cr, z2_cr = 200, 5000


# #### Total Gradient Amplitude
#From Fatiando
TGA = transform.tga(xi, yi, tf, shape, method = 'fd')


fig=plt.figure(figsize=(18,4.8))

plt.subplot(1,3,1)
plt.title('(a)',fontsize=14,loc='center')
rect1 = patches.Rectangle((5.,5.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect2 = patches.Rectangle((y1_L/1000.,x1_L/1000.),(y2_L-y1_L)/1000.,(x2_L-x1_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect3 = patches.Rectangle((y3_L/1000.,x3_L/1000.),(y4_L-y3_L)/1000.,(x4_L-x3_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect4 = patches.Rectangle((y1_c/1000.,x1_c/1000.),(y2_c-y1_c)/1000.,(x2_c-x1_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect5 = patches.Rectangle((y3_c/1000.,x3_c/1000.),(y4_c-y3_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect6 = patches.Rectangle((y1_cr/1000.,x1_cr/1000.),(y2_cr-y1_cr)/1000.,(x2_cr-x1_cr)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect7 = patches.Rectangle((y5_c/1000.,x3_c/1000.),(y6_c-y5_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')

im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,tf.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT',labelpad=-21,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(9.5,10,'P1',color='w',weight='bold')
plt.text(11,32,'P2',color='w',weight='bold')
plt.text(25,25.5,'P3',color='w',weight='bold')
plt.text(24,10,'P4',color='w',weight='bold')

plt.subplot(1,3,2)
plt.title('(b)',fontsize=14,loc='center')
rect1 = patches.Rectangle((5.,5.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect2 = patches.Rectangle((y1_L/1000.,x1_L/1000.),(y2_L-y1_L)/1000.,(x2_L-x1_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect3 = patches.Rectangle((y3_L/1000.,x3_L/1000.),(y4_L-y3_L)/1000.,(x4_L-x3_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect4 = patches.Rectangle((y1_c/1000.,x1_c/1000.),(y2_c-y1_c)/1000.,(x2_c-x1_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect5 = patches.Rectangle((y3_c/1000.,x3_c/1000.),(y4_c-y3_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect6 = patches.Rectangle((y1_cr/1000.,x1_cr/1000.),(y2_cr-y1_cr)/1000.,(x2_cr-x1_cr)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect7 = patches.Rectangle((y5_c/1000.,x3_c/1000.),(y6_c-y5_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')

im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,TGA.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT/m',labelpad=-15,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(9.5,10,'P1',color='w',weight='bold')
plt.text(11,32,'P2',color='w',weight='bold')
plt.text(25,25.5,'P3',color='w',weight='bold')
plt.text(24,10,'P4',color='w',weight='bold')

plt.subplot(1,3,3)
plt.title('(c)',fontsize=14,loc='center')
rect1 = patches.Rectangle((5.,5.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect2 = patches.Rectangle((y1_L/1000.,x1_L/1000.),(y2_L-y1_L)/1000.,(x2_L-x1_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect3 = patches.Rectangle((y3_L/1000.,x3_L/1000.),(y4_L-y3_L)/1000.,(x4_L-x3_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect4 = patches.Rectangle((y1_c/1000.,x1_c/1000.),(y2_c-y1_c)/1000.,(x2_c-x1_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect5 = patches.Rectangle((y3_c/1000.,x3_c/1000.),(y4_c-y3_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect6 = patches.Rectangle((y1_cr/1000.,x1_cr/1000.),(y2_cr-y1_cr)/1000.,(x2_cr-x1_cr)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect7 = patches.Rectangle((y5_c/1000.,x3_c/1000.),(y6_c-y5_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')

im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,amplitude.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT',labelpad=-20,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(9.5,10,'P1',color='w',weight='bold')
plt.text(11,32,'P2',color='w',weight='bold')
plt.text(25,25.5,'P3',color='w',weight='bold')
plt.text(24,10,'P4',color='w',weight='bold')


plt.subplots_adjust(wspace=0.25)
plt.savefig('FIG_4.png',dpi=300,bbox_inches='tight')
plt.close('all')
gc.collect()




fig=plt.figure(figsize=(18,4.8))

plt.subplot(1,3,1)
plt.title('(a)',fontsize=14,loc='center')
rect1 = patches.Rectangle((5.,5.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect2 = patches.Rectangle((y1_L/1000.,x1_L/1000.),(y2_L-y1_L)/1000.,(x2_L-x1_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect3 = patches.Rectangle((y3_L/1000.,x3_L/1000.),(y4_L-y3_L)/1000.,(x4_L-x3_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect4 = patches.Rectangle((y1_c/1000.,x1_c/1000.),(y2_c-y1_c)/1000.,(x2_c-x1_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect5 = patches.Rectangle((y3_c/1000.,x3_c/1000.),(y4_c-y3_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect6 = patches.Rectangle((y1_cr/1000.,x1_cr/1000.),(y2_cr-y1_cr)/1000.,(x2_cr-x1_cr)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect7 = patches.Rectangle((y5_c/1000.,x3_c/1000.),(y6_c-y5_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')

im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,predict.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT',labelpad=-21,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(9.5,10,'P1',color='w',weight='bold')
plt.text(11,32,'P2',color='w',weight='bold')
plt.text(25,25.5,'P3',color='w',weight='bold')
plt.text(24,10,'P4',color='w',weight='bold')

# Tweak spacing to prevent clipping of ylabel

plt.subplot(1,3,2)
plt.title('(b)',fontsize=14,loc='center')
rect1 = patches.Rectangle((5.,5.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect2 = patches.Rectangle((y1_L/1000.,x1_L/1000.),(y2_L-y1_L)/1000.,(x2_L-x1_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect3 = patches.Rectangle((y3_L/1000.,x3_L/1000.),(y4_L-y3_L)/1000.,(x4_L-x3_L)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect4 = patches.Rectangle((y1_c/1000.,x1_c/1000.),(y2_c-y1_c)/1000.,(x2_c-x1_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect5 = patches.Rectangle((y3_c/1000.,x3_c/1000.),(y4_c-y3_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect6 = patches.Rectangle((y1_cr/1000.,x1_cr/1000.),(y2_cr-y1_cr)/1000.,(x2_cr-x1_cr)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
rect7 = patches.Rectangle((y5_c/1000.,x3_c/1000.),(y6_c-y5_c)/1000.,(x4_c-x3_c)/1000.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')

im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,(tf.reshape(shape)-predict.reshape(shape)), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT',labelpad=-21,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(9.5,10,'P1',color='w',weight='bold')
plt.text(11,32,'P2',color='w',weight='bold')
plt.text(25,25.5,'P3',color='w',weight='bold')
plt.text(24,10,'P4',color='w',weight='bold')

plt.subplot(1,3,3)
plt.title('(c)',fontsize=14,loc='center')

residuo=tf.reshape(shape)-predict.reshape(shape)
mu_res=np.mean(residuo)
sigma_res=np.std(residuo)
n, bins, patches = plt.hist(residuo.ravel(), 70, normed=1, facecolor='blue', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu_res, sigma_res)
plt.plot(bins, y, 'r--')
plt.xlabel('Residuals (nT)', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)
plt.tick_params(labelsize=13)
plt.axis([-8, 8, 0,  1])
plt.text(3,0.7,'$\mu=%1.2f$'%(np.abs(mu_res)),color='k',fontsize=18)
plt.text(3,0.6,'$\sigma=%1.2f$'%(sigma_res),color='k',fontsize=18)
plt.grid(True)

plt.subplots_adjust(wspace=0.25)
plt.savefig('FIG_4b.png',dpi=300,bbox_inches='tight')
plt.close('all')
gc.collect()