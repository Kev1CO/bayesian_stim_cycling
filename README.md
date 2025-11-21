# Bayesian stimulation hand cycling project

## Required installation
- Python==3.9 (for the stimulator)

### Stimulator:
- pysciencemode: conda install -c conda-forge pysciencemode (https://github.com/s2mLab/pyScienceMode)

### Encoder:
- nidaqmx: conda install conda-forge::nidaqmx-python (https://anaconda.org/conda-forge/nidaqmx-python)
- NI-DAQ software and drivers: https://www.ni.com/fr/support/downloads/drivers/download.ni-daq-mx.html?srsltid=AfmBOoq5Z4j-iU1ba810SYTwTJGMpS7VuC-yRcFi3tORrE3IQoFDrhIf#577117

### Optimization:
- Bioptim: Using the last bioptim commit (PR#1033) (https://github.com/pyomeca/bioptim)
- skopt: conda install conda-forge::scikit-optimize (https://anaconda.org/conda-forge/scikit-optimize)