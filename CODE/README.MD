# Amplitude of the magnetic anomaly vector in the interpretation of total-field anomaly data at low magnetic latitudes

by Felipe F. Melo, Shayane P. Gonzalez, Valéria C. F. Barbosa and Vanderlei C. Oliveira Jr (Observatório Nacional) (2019)

Email: felipe146@hotmail.com (F.F. Melo)

This repository contains a collection of Python codes and synthetic data 
example for a paper about the open-source package to compute the aplitude of the magnetic anomaly vector .The 
example data reproduce the synthetc results and figures shown in the publication.
The python package is compatible with Python 2.7 programming language. To run the programs, the numpy, 
matplotlib and Fatiando a Terra library are required. 

1 - Abstract

2 - Content

3 - Prerequisites

4 - Parameterization

5 - Running the files

6 - License

===============================================================================


1 - Abstract
----------------------
We propose the use of the amplitude of the magnetic vector for qualitative interpretation in large areas at low magnetic latitudes. The amplitude data is weakly dependent on the magnetization direction. Hence, the amplitude data require no prior knowledge of the source magnetization direction. The amplitude data produce maxima over the causative sources, allowing the definition of the horizontal projections of the sources. This characteristic is attractive for interpretation at low magnetic latitudes because at these regions the interpretation is not straightforward as the magnetic field increases the complexity of the anomalies. We compute the amplitude data using the equivalent-layer technique to transform the total-field anomaly data into the three orthogonal components of the magnetic anomaly vector. We analyze the results of tests in synthetic data simulating a main geomagnetic field at high, mid and low latitudes, with sources ranging from compact to elongated forms, including a dipping source. These sources, that give rise to the simulated anomalies, have both induced and strong remanent magnetizations. By comparing the amplitude data with the total gradient, we show that the amplitude data delineate the boundaries of the sources in a better way. We apply both the amplitude data and the total gradient to real total-field anomaly data on a large area in the Amazonian Craton, Brazil, at low magnetic latitudes. The amplitude data show a better performance in delineating geologic bodies being consistent with the outcropping intrusions in the geologic map. Moreover, the amplitude data revealed new geologic bodies that were not present in the geologic map. The clear alignment of these new bodies with the outcropping intrusions suggested the continuity of these intrusions in depth. This result is a step forward in understanding this area, which has a poor geological knowledge. Hence, the amplitude data can provide an apparent geologic map especially in areas at low latitudes with remanent magnetized bodies.

2 - Content
----------------------

The synthetic tests can be reproduced with these codes. The tests can be reproduced from the folders
`High_I_D`, `Mid_I_D` and `Low_I_D`.

Python codes:

	- X.py: General Python module containing the functions to compute the forward model,
    the equivalent layer, estimate the parameters, compute the three orthogonal vectors and
    the amplitude. The "X" stands for the latitude of the test, eg., "High_I_D", "Mid_I_D" and "Low_I_D".
	
	- X_plot.py: Plot the results.
    The "X" stands for the latitude of the test, eg., "High_I_D", "Mid_I_D" and "Low_I_D".
	
Outputs: 
 
	- figures - all figures that generated the synthetic example in the manuscript will be saved.
  
  	- input_mag.dat 
		Synthetic magnetic data generated using the Python packaged "Fatiando a Terra":	
		http://fatiando.org/. This data is an example used in the current publication.
	
  	- parameters.dat 
		The parameters estimated with the equivalent layer.

  	- predict.dat 
		The predicted data, from the total field-anomaly, with the equivalent layer.

  	- amplitude.dat 
		The amplitude of the magnetic anomaly vector.

3 - Prerequisites
----------------------
To run the programs, the "numpy", "matplotlib" and "Fatiando a Terra" library are required.  
The easier way to get Python and all libraries installed is through the Anaconda Python 
distribution (https://www.anaconda.com/distribution/). After installed Anaconda, install the libraries 
by running the following command in your terminal:

	conda install numpy matplotlib
  
The version of the "Fatiando a Terra" library is available at: https://www.fatiando.org/dev/install.html#installing-the-latest-development-version

The programs are compatible with both Python 2.7.
 
4 - Parameterization
----------------------
There are the two parameters that need to be defined: the depth of the equivalent layer and the regularization parameter.

5 - Running the files
----------------------
After Anaconda and the libraries are installed, you need to run 
the code `High_I_D.py` to generate the results and `High_I_D_plot.py` to generated the figures.
The programs are distinct because of the memory consuming issue of the equivalent layer.
There are two easy ways to run the synthetic tests:

1. Via command line: 

	1.1 On linux based systems open the command prompt windows (terminal), and run the command:
	
	On Python 2.7: python High_I_D_plot.py 
	On Python 3.7: python3 High_I_D_plot.py 

	1.2 On windows based systems open the Iphyton console (Python 2.7) or 
	Anaconda Navigator and then Jupyter QtConsole (Python 3.7), and run the command:

	run High_I_D_plot.py

2. Open the Spyder program that comes with the Ananconda installation. In the Spyder, open
the example script "High_I_D_plot.py" and click in "play" inside the Spyder to execute it.

If everything runs ok, the figures shown in the publication will be saved in the same folder of the code.

The file `High_I_D.py` is a Python-based command line script. By using this 
script, the figures of the synthetic test presented in the publication can be reproduced.
The figures generated may differ from the publication and can be adapted in the script `High_I_D_plot.py`.

6 - License
----------------------
The following legal note is restricted solely to the content of the named files. It cannot
overrule licenses from the Python standard distribution modules, which are imported and
used therein.

All the files are distributed under the following license agreement:

BSD 3-clause license

Copyright (c) 2019 Felipe F. Melo and Valéria C.F. Barbosa.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of any contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
