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


Output:
    input_mag.dat    
"""
from __future__ import division
import numpy as np
from fatiando import mesher, gridder, utils
from fatiando.gravmag import prism

shape = (116, 97)
area = (0, 34800, 0, 29100)
xi, yi, zi = gridder.regular(area, shape, z = -150)

#Set the inclination and declination of the regional field
inc_o, dec_o = -8., -20.  
# Set the inclination and declination of the source
inc_s, dec_s = -8., -20.  

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
#Magnetization intensity
mag_m = 7


#L shape source
x1_L, x2_L = 26000, 30000
y1_L, y2_L = 8000, 8500
#Top and bottom of 1st the prism
z1_L, z2_L = 200, 1100

x3_L, x4_L = 29500, 30000
y3_L, y4_L = 8500, 10500
#Top and bottom of 2nd the prism
z3_L, z4_L = 200, 1100
#Magnetization intensity
mag_L = 3

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
#Magnetization intensity
mag_c = 1.5
inc_c = 45.
dec_c = 45.

#vertical prism
x1_cr, x2_cr = 11000, 13000
y1_cr, y2_cr = 19000, 21000
z1_cr, z2_cr = 200, 5000
#Magnetization intensity
mag_cr = 1.5
inc_cr = 60.
dec_cr = 60.

#From Fatiando a Terra
# Compute the cosine directions of the main geomagetic field (F)
F = utils.ang2vec(1,inc_o, dec_o)

#Generate a model From Fatiando a Terra
model_mag = [
    mesher.Prism(x1_cr,x2_cr,y1_cr,y2_cr,z1_cr,z2_cr,{'magnetization': utils.ang2vec(mag_cr, inc_cr, dec_cr)}),
    mesher.Prism(x1_c,x2_c,y1_c,y2_c,z1_c,z2_c,{'magnetization': utils.ang2vec(mag_c, inc_c, dec_c)}),
    mesher.Prism(x3_c,x4_c,y3_c,y4_c,z3_c,z4_c,{'magnetization': utils.ang2vec(mag_c, inc_c, dec_c)}),
    mesher.Prism(x3_c,x4_c,y5_c,y6_c,z3_c,z4_c,{'magnetization': utils.ang2vec(mag_c, inc_c, dec_c)}),
    mesher.Prism(x1_L,x2_L,y1_L,y2_L,z1_L,z2_L,{'magnetization': utils.ang2vec(mag_L, inc_s, dec_s)}),
    mesher.Prism(x3_L,x4_L,y3_L,y4_L,z3_L,z4_L,{'magnetization': utils.ang2vec(mag_L, inc_s, dec_s)}),    
    mesher.Prism(x1,x2,y1m[0],y2m[0],zo_t[0],zo_b[0],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[1],y2m[1],zo_t[1],zo_b[1],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[2],y2m[2],zo_t[2],zo_b[2],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[3],y2m[3],zo_t[3],zo_b[3],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[4],y2m[4],zo_t[4],zo_b[4],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[5],y2m[5],zo_t[5],zo_b[5],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[6],y2m[6],zo_t[6],zo_b[6],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[7],y2m[7],zo_t[7],zo_b[7],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[8],y2m[8],zo_t[8],zo_b[8],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[9],y2m[9],zo_t[9],zo_b[9],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[10],y2m[10],zo_t[10],zo_b[10],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[11],y2m[11],zo_t[11],zo_b[11],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[12],y2m[12],zo_t[12],zo_b[12],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[13],y2m[13],zo_t[13],zo_b[13],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[14],y2m[14],zo_t[14],zo_b[14],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[15],y2m[15],zo_t[15],zo_b[15],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[16],y2m[16],zo_t[16],zo_b[16],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[17],y2m[17],zo_t[17],zo_b[17],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[18],y2m[18],zo_t[18],zo_b[18],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[19],y2m[19],zo_t[19],zo_b[19],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[20],y2m[20],zo_t[20],zo_b[20],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[21],y2m[21],zo_t[21],zo_b[21],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[22],y2m[22],zo_t[22],zo_b[22],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[23],y2m[23],zo_t[23],zo_b[23],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[24],y2m[24],zo_t[24],zo_b[24],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[25],y2m[25],zo_t[25],zo_b[25],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[26],y2m[26],zo_t[26],zo_b[26],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[27],y2m[27],zo_t[27],zo_b[27],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[28],y2m[28],zo_t[28],zo_b[28],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[29],y2m[29],zo_t[29],zo_b[29],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[30],y2m[30],zo_t[30],zo_b[30],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[31],y2m[31],zo_t[31],zo_b[31],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[32],y2m[32],zo_t[32],zo_b[32],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[33],y2m[33],zo_t[33],zo_b[33],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[34],y2m[34],zo_t[34],zo_b[34],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[35],y2m[35],zo_t[35],zo_b[35],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[36],y2m[36],zo_t[36],zo_b[36],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[37],y2m[37],zo_t[37],zo_b[37],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[38],y2m[38],zo_t[38],zo_b[38],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)}),
    mesher.Prism(x1,x2,y1m[39],y2m[39],zo_t[39],zo_b[39],{'magnetization': utils.ang2vec(mag_m, inc_s, dec_s)})]


#total field from Fatiando a Terra
tf = utils.contaminate(prism.tf(xi, yi, zi, model_mag,inc_o, dec_o),0.0001,
                       percent=True)


#save for the plot
out=np.array([yi,xi,tf])        
out=out.T
np.savetxt('input_mag.dat',out,delimiter=' ',fmt='%1.8f')
out = None
