# Synthetic test 1 – Simulated field at high latitude and induced magnetization

Running the code `generate_input.py` will generate the input data and the code
`High_I_D.py` will allow the reprodution of the results.
The figures generated may differ from the publication and 
can be adapted in the script `High_I_D_plot.py`.

## Input:

- input_data.dat - is generated in the file `generate_input.py`

## Parameters:

- Depth of the equivalent sources:    
    Zj - a float number. 
                                  
- Regularizing parameter:
    lambida - a float number. 

## Outputs:

- input_mag.dat
    The input data generated
    
- predict.dat
    The predicted data.
    
- amplitude.dat
    Amplitude of the anomaly vector.

- figures:
    Fig_2.png (a) Total-field anomaly. (b) Total-gradient. (c) Amplitude of the magnetic anomaly vector.
    (d) Predicted data. (e) Residuals. (f) Histogram of the residuals.
					 
