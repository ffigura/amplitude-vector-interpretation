"""
Synthetic test 1

A Python program to plot the results of the 
Synthetic test 1 - Simulated field at high latitude and induced magnetization

This code is released from the paper: 
Amplitude of the magnetic anomaly vector in the interpretation
 of total-field anomaly data at low magnetic latitudes

The program is under the conditions terms in the file README.txt

authors: Felipe F. Melo and Shayane P. Gonzalez, 2019
email: felipe146@hotmail.com, shayanegonzalez@gmail.com 

Input:
    - input_mag.dat
        The input data generated from generate_input.py

    - data\parameters.dat 
    	The parameters estimated with the equivalent layer.

    - data\predict.dat 
    	The predicted data, from the total field-anomaly, with the equivalent
        layer.

    - data\B_eq.dat 
    	The amplitude of the magnetic anomaly vector. 
        
    - L_curve.dat
       The regularization parameters, norm of the residuals and of the model

Output:
    Figures: 
        Fig_4.png (a) Total-field anomaly. (b) Total-gradient. 
        (c) Amplitude of the magnetic anomaly vector. 
        (d) L-curve. (e) Residuals. (f) Histogram of the residuals.

        parameters.png The estimated parameters
        
        predict.png The predicted data 
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from fatiando.gravmag import transform
import matplotlib.patches as patches
import gc
import matplotlib.mlab as mlab

shape = (100, 110)

input_data=np.loadtxt('input_mag.dat')

amplitude_input=np.loadtxt('data\B_eq_1e-15.dat')

predict_input=np.loadtxt('data\predict_1e-15.dat')

parameters_input=np.loadtxt('data\parameters_1e-15.dat')

yi=input_data[:,0]
xi=input_data[:,1]
zi=input_data[:,2]
tf=input_data[:,3]
amplitude=amplitude_input[:,3]
predict=predict_input[:,3]
parameters=parameters_input[:,3]

Lcurve_input=np.loadtxt('L_curve.dat')
regul=Lcurve_input[:,0]
phi_list=Lcurve_input[:,1]
p_list=Lcurve_input[:,2]

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
x1, x2=3000,7000

#L shape source
x1_L, x2_L = 13000, 17000
y1_L, y2_L = 3000, 3500
#Top and bottom of 1st the prism
z1_L, z2_L = 200, 1100

x3_L, x4_L = 16500, 17000
y3_L, y4_L = 3500, 5500
#Top and bottom of 2nd the prism
z3_L, z4_L = 200, 1100

#three sources
# 1st prism
x1_c, x2_c = 13000, 17000
y1_c, y2_c = 11000, 19000
#Top and bottom of 1st the prism
z1_c, z2_c = 1000, 5000
# 2nd prism
x3_c, x4_c = 14500, 15500
y3_c, y4_c = 16500, 17500
#Top and bottom of 2nd the prism
z3_c, z4_c = 200, 1000
# 3rd prism 
y5_c, y6_c = 12500, 13500

#vertical prism
x1_cr, x2_cr = 4000, 6000
y1_cr, y2_cr = 14000, 16000
z1_cr, z2_cr = 200, 5000


# #### Total Gradient Amplitude
#From Fatiando
TGA = transform.tga(xi, yi, tf, shape, method = 'fd')

fig=plt.figure(figsize=(5,4))
im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,parameters.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('$A.m^{2}$',labelpad=-21,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.yaxis.set_ticks([0,5,10,15,20])
plt.savefig('parameters.png',dpi=300,bbox_inches='tight')
plt.close('all')
gc.collect()

fig=plt.figure(figsize=(5,4))
im=plt.contourf(yi.reshape(shape)/1000.,xi.reshape(shape)/1000.,predict.reshape(shape), 30, cmap='jet')
ax=plt.gca()
ax.set_ylabel('Northing (km)', fontsize = 14)
ax.set_xlabel('Easting (km)', fontsize = 14)
ax.tick_params(labelsize=13)
cbar=fig.colorbar(im,pad=0.01,shrink=1)
cbar.set_label('nT',labelpad=-30,y=-0.03, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.yaxis.set_ticks([0,5,10,15,20])
plt.savefig('predict.png',dpi=300,bbox_inches='tight')
plt.close('all')
gc.collect()


fig=plt.figure(figsize=(17,9))

plt.subplot(2,3,1)
plt.title('(a)',fontsize=14,loc='center')
rect1 = patches.Rectangle((vety[0]/1000.,x1/1000.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
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
cbar.set_label('nT',labelpad=-33,y=-0.04, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(1,5,'P1',color='w',weight='bold')
plt.text(7,17,'P2',color='w',weight='bold')
plt.text(20,17,'P3',color='w',weight='bold')
plt.text(19,5,'P4',color='w',weight='bold')
ax.yaxis.set_ticks([0,5,10,15,20])

plt.subplot(2,3,2)
plt.title('(b)',fontsize=14,loc='center')
rect1 = patches.Rectangle((vety[0]/1000.,x1/1000.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
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
cbar.set_label('nT/m',labelpad=-17,y=-0.04, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(1,5,'P1',color='w',weight='bold')
plt.text(7,17,'P2',color='w',weight='bold')
plt.text(20,17,'P3',color='w',weight='bold')
plt.text(19,5,'P4',color='w',weight='bold')
ax.yaxis.set_ticks([0,5,10,15,20])

plt.subplot(2,3,3)
plt.title('(c)',fontsize=14,loc='center')
rect1 = patches.Rectangle((vety[0]/1000.,x1/1000.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
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
cbar.set_label('nT',labelpad=-23,y=-0.04, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(1,5,'P1',color='w',weight='bold')
plt.text(7,17,'P2',color='w',weight='bold')
plt.text(20,17,'P3',color='w',weight='bold')
plt.text(19,5,'P4',color='w',weight='bold')
ax.yaxis.set_ticks([0,5,10,15,20])

plt.subplot(2,3,4)
plt.title('(d)',fontsize=14,loc='center')
plt.plot(phi_list,p_list,'-o')
plt.plot(phi_list[5],p_list[5],'ro')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$log||G\hat{p}(\lambda)-d||_{2}$', fontsize = 14) 
plt.ylabel('$log||\hat{p}(\lambda)||_{2}$', fontsize = 14)
plt.tick_params(labelsize=13)# -*- coding: utf-8 -*-
plt.text(1e-1,1e19,'$\lambda=1e-15$',color='k',weight='bold',fontsize=18)
plt.grid(True)

plt.subplot(2,3,5)
plt.title('(e)',fontsize=14,loc='center')
rect1 = patches.Rectangle((vety[0]/1000.,x1/1000.),4.,4.,linewidth=1,edgecolor='black',linestyle='-',facecolor='none')
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
cbar.set_label('nT',labelpad=-30,y=-0.04, rotation=0,fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)
ax.add_patch(rect6)
ax.add_patch(rect7)
plt.text(1,5,'P1',color='k',weight='bold')
plt.text(7,17,'P2',color='k',weight='bold')
plt.text(20,17,'P3',color='k',weight='bold')
plt.text(19,5,'P4',color='k',weight='bold')
ax.yaxis.set_ticks([0,5,10,15,20])

plt.subplot(2,3,6)
plt.title('(f)',fontsize=14,loc='center')

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
plt.axis([-3, 3, 0,  1])
plt.text(1,0.7,'$\mu=%1.2f$'%(np.abs(mu_res)),color='k',fontsize=18)
plt.text(1,0.6,'$\sigma=%1.2f$'%(sigma_res),color='k',fontsize=18)
plt.grid(True)

plt.subplots_adjust(wspace=0.3,hspace=0.35)
plt.savefig('FIG_2.png',dpi=300,bbox_inches='tight')
plt.close('all')
gc.collect()