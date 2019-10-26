"""
Synthetic test 1

A Python program to generate the Amplitude of the magnetic anomaly vector
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

Parameters:
    - Depth of the equivalent sources:    
        Zj - a float number. 
                                  
    - Regularizing parameter:
        lambida - a float number. 

Output:
    - predict.dat 
    	The predicted data, from the total field-anomaly, with the equivalent
        layer.

    - amplitude.dat 
    	The amplitude of the magnetic anomaly vector. 
"""
from __future__ import division
import numpy as np
from fatiando import utils
from fatiando.gravmag import sphere
from fatiando.mesher import PointGrid


shape = (116, 97)
area = (0, 34800, 0, 29100)

input_data=np.loadtxt('input_mag.dat')

yi=input_data[:,0]
xi=input_data[:,1]
tf=input_data[:,2]
zi=np.zeros_like(xi) #flight height

#Set the inclination and declination of the regional field
inc_o, dec_o = 90., 0.  


# Compute the cosine directions of the main geomagetic field (F)
F = utils.ang2vec(1,inc_o, dec_o)

#This is for the equivalent layer
shapej=shape
areaj=area
zj = 900
N = shape[0]*shape[1]
M = shapej[0]*shapej[1]

#From Fatiando
layer = PointGrid(areaj, zj, shapej)


G = np.empty((N,M),dtype =float)
for i, c in enumerate(layer):
    #From Fatiando
    G[:,i] = sphere.tf(xi, yi, zi, [c], inc_o, dec_o, pmag = F) 
GTG = np.empty((M,M),dtype =float)
GTG = np.dot(G.T, G)
GTd = np.dot(G.T,tf)

# Estimation of parameter p (overdetermined)
lambida=0
I = np.identity(M)

p = np.linalg.solve(GTG + lambida*I, GTd)

I = None
GTd = None
GTG = None
predict=np.dot(G,p)

#Predict data
out=[]
predict=np.dot(G,p)
out=np.array([yi,xi,predict])        
out=out.T
np.savetxt('predict.dat',out,delimiter=' ',fmt='%1.8f')

out = None
G = None

# Compute the transformed matrix
Tx = np.empty((N,M),dtype =float)
Ty = np.empty((N,M),dtype =float)
Tz = np.empty((N,M),dtype =float)
for i, c in enumerate(layer):
    Tx[:,i] = sphere.bx(xi, yi, zi, [c], pmag = F)
    Ty[:,i] = sphere.by(xi, yi, zi, [c], pmag = F)
    Tz[:,i] = sphere.bz(xi, yi, zi, [c], pmag = F)

# compute the components
Bx_eq = np.dot(Tx,p)
By_eq = np.dot(Ty,p)
Bz_eq = np.dot(Tz,p)

Tx = None
Ty = None
Tz = None
p = None

#Amplitude of the magnetic anomaly vector
B_eq = np.sqrt(Bx_eq*Bx_eq + By_eq*By_eq + Bz_eq*Bz_eq)

Bx_eq = None
By_eq = None
Bz_eq = None

out=np.array([yi,xi,B_eq])        
out=out.T
np.savetxt('amplitude.dat',out,delimiter=' ',fmt='%1.8f')
out = None
B_eq = None