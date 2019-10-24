# Synthetic test 3 – Simulated field at low latitude and remanent magnetization

Running the code `Mid_I_D.py` will allow the reprodution of the results.
The figures generated may differ from the publication and 
can be adapted in the script `Mid_I_D_plot.py`.

## Input:

- input - input_data.dat (is generated in the main file)

## Parameters:

- Depth of the equivalent sources:    
    Zj - a float number. 
                                  
- Regularization parameter:
    lambda - a float number. 

## Outputs:

- input_mag.dat
    The input data generated
    
- parameters.dat
    The estimated parameters
    
- predict.dat
    The predicted data.
    
- amplitude.dat
    Amplitude of the anomaly vector.

- figures:
    Fig_4.png (a) Total-field anomaly. (b) Total-gradient. (c) Amplitude of the magnetic anomaly vector.
    Fig_4b.png (a) Predicted data. (b) Residuals. (c) Histogram of the residuals.
				
