"""
Synthetic test 2

A Python program to generate the Amplitude of the magnetic anomaly vector
Synthetic test 2 - Simulated field at mid-latitude and remanent magnetization

This code is released from the paper: 
Amplitude of the magnetic anomaly vector in the interpretation
 of total-field anomaly data at low magnetic latitudes

The program is under the conditions terms in the file README.txt

authors: Felipe F. Melo and Shayane P. Gonzalez, 2019
email: felipe146@hotmail.com, shayanegonzalez@gmail.com 

Input:
    - input_mag.dat
        The input data generated from generate_input.py (y,x,z,tf)

Parameters:
    - Depth of the equivalent sources:    
        zj - a float number. 
                                  
    - Regularizing parameter:
        lambida - a float number. 

Output:
    - parameters.dat 
    	The parameters estimated with the equivalent layer.

    - predict.dat 
    	The predicted data, from the total field-anomaly, with the equivalent
        layer.

    - B_eq.dat 
    	The amplitude of the magnetic anomaly vector. 
        
    - L_curve.dat
        array with the regularization parameters, function phi and function p 
"""
from __future__ import division
import numpy as np
from fatiando import utils
from fatiando.gravmag import sphere
from fatiando.mesher import PointGrid


shape = (100, 110)
area = (0, 20000, 0, 22000)

input_data=np.loadtxt('input_mag.dat')

yi=input_data[:,0]
xi=input_data[:,1]
zi=input_data[:,2]
tf=input_data[:,3] #flight height

#Set the inclination and declination of the regional field
inc_o, dec_o = 45., 45.  

# Compute the cosine directions of the main geomagetic field (F)
F = utils.ang2vec(1,inc_o, dec_o)

#This is for the equivalent layer
shapej=shape
areaj=area
zj = 500
N = shape[0]*shape[1]
M = shapej[0]*shapej[1]

regul=np.logspace(-20,-11,10)

#From Fatiando
layer = PointGrid(areaj, zj, shapej)

#to the L curve plot
phi_list=[]
p_list=[]

G = np.empty((N,M),dtype =float)
for i, c in enumerate(layer):
    #From Fatiando
    G[:,i] = sphere.tf(xi, yi, zi, [c], inc_o, dec_o, pmag = F) 

for i,lambida in enumerate(regul): 
    
    lambida=float(format(lambida, '.3e'))

    print '\niteration %d lambida=%1.3e \n'%(i,lambida)
#   mu=1e-20
    I = np.identity(M)
    GTG = np.dot(G.T, G)
    GTd = np.dot(G.T,tf.ravel())

    p = np.linalg.solve(GTG + lambida*I, GTd)

    I = None
    GTd = None
    GTG = None
    predict=np.dot(G,p)
    
    out=[]
    out=np.array([yi,xi,zi,p])        
    out=out.T
    np.savetxt('data\parameters_'+str(lambida)+'.dat',out,
               delimiter=' ',fmt='%1.3f')
    out = None
    
    pnorm=np.sum(p*p)
    p_list.append(pnorm)
    
    #Predict data
    out=[]
    predict=np.dot(G,p)
    res=(tf.ravel()-predict)
    phi=np.sum(res*res)
    phi_list.append(phi)
    
    out=np.array([yi,xi,zi,predict])        
    out=out.T
    np.savetxt('data\predict_'+str(lambida)+'.dat',out,
               delimiter=' ',fmt='%1.3f')
    
    out = None
    
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
    
    out=np.array([yi,xi,zi,B_eq])        
    out=out.T
    np.savetxt('data\B_eq_'+str(lambida)+'.dat',out,
               delimiter=' ',fmt='%1.3f')
    out = None
    B_eq = None
    
out=np.array([regul,np.array(phi_list),np.array(p_list)])
out=out.T    
np.savetxt('L_curve.dat',out,delimiter=' ',fmt='%1.3e')
out=None