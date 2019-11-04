# Amplitude of the magnetic anomaly vector in the interpretation of total-field anomaly data at low magnetic latitudes

by
Felipe F. Melo, Shayane P. Gonzalez, Valéria C. F. Barbosa and Vanderlei C. Oliveira Jr

## About

This paper has been submitted for publication in the journal *Journal of Applied Geophysics*.

This repository contains the source code to perform the synthetic tests presented. The `generate_input.py` generates the synthetic data, the codes `High_I_D.py`, `Mid_I_D.py` and `Low_I_D.py` compute the amplitude of the magnetic anomaly vector andthe codes `High_I_D_plot.py`, `Mid_I_D_plot.py` and `Low_I_D_plot.py` generate the figures of the synthetic test.

The programs are compatible with Python 2.7 programming language.
 
## Abstract

We propose the use of the amplitude of the magnetic vector (amplitude data) for qualitative interpreting large areas at low magnetic latitudes. The amplitude data are weakly dependent on the magnetization direction. Hence, the amplitude data require no prior knowledge of the source magnetization direction. The amplitude data produce maxima over the causative sources, allowing the definition of the horizontal projections of the sources. This characteristic is attractive for interpretation at low magnetic latitudes because at these regions the interpretation of the total-field anomaly is not straightforward. We compute the amplitude data using the equivalent-layer technique to transform the total-field anomaly data into the three orthogonal components of the magnetic anomaly vector. We analyze the results of tests in synthetic data simulating a main geomagnetic field at high, mid and low latitudes, with sources ranging from compact to elongated forms, including a dipping source. These sources, that give rise to the simulated anomalies, have both induced and strong remanent magnetizations. By comparing the amplitude data with the total gradient, we show that the amplitude data delineate the boundaries of the sources in a better way. We apply both the amplitude data and the total gradient to a real total-field anomaly over a large area of the Amazonian Craton, northern Brazil, located at low magnetic latitudes. The amplitude data show a better performance in delineating geologic bodies being consistent with the outcropping intrusions in the geologic map. Moreover, the amplitude data revealed new geologic bodies that were not present in the geologic map. The clear alignment of these new bodies with the outcropping intrusions suggested the continuity of these intrusions in depth. This result is a step forward in understanding this area, which has a poor geological knowledge. Hence, the amplitude data can provide an apparent-geologic map especially in areas at low latitudes with remanent magnetized bodies.   

## Content

- generate_input.py:
	Python code to compute the forward model.

- X_I_D.py:
	General Python module containing the functions to compute the equivalent layer, estimate the parameters, the three orthogonal vectors and the amplitude. The "X" stands for the latitude of the test, eg., "High", "Mid" or "Low".
	
- X_I_D_plot.py:
	Plot the results.
	
Test data:

- input_mag.dat:
	Synthetic total-field anomaly data generated using the Python packaged
	"Fatiando a Terra": http://fatiando.org/. This data is an example used
	in the current publication.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/ffigura/amplitude-vector-interpretation.git
    
or [download a zip archive](https://github.com/ffigura/amplitude-vector-interpretation/archive/master.zip).


## Dependencies

The Python programs for compute the syntehtic data - "generate_input.py" - and the amplitude of the magnetic anomaly vector - "X_I_D.py" - require the Python library "numpy" and the last development version of the free Python packaged "Fatiando a Terra" ( https://www.fatiando.org/dev/install.html#installing-the-latest-development-version ). The script "X_plot.py" requires the same Python packages in addtion to "matplotlib". 

The easier way to get Python and all libraries installed is through the Anaconda Python 
distribution (https://www.anaconda.com/distribution/). After installed Anaconda, install the libraries 
by running the following command in your terminal:

	conda install numpy matplotlib

The programs are compatible with Python 2.7.

## Reproducing the results

The results and figures for the synthetic test are reproducible from the folders `/High_I_D`, `/Mid_I_D` and `/Low_I_D`.
Running the code `generate_input.py` will generate the synthetic data, the code `X_I_D.py` will allow the reprodution of the results and the code `X_I_D_plot.py` will generate the figures. For more information read the file `README.MD` or `README.txt` in the folder `/code`.

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
*Journal of Applied Geophysics*.
