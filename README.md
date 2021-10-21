# Cell Membrane-Virus Interaction

Code for the paper [FIB-SEM as a Volume Electron Microscopy Approach to Study Cellular Architectures in SARS-CoV-2 and Other Viral Infections: A Practical Primer for a Virologist](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8066521/).

## Requirements

The following python packages need to be installed before running the script (with pip or conda):

- numpy
- scipy
- trimesh
- pandas
- open3d
- seaborn

## Raw Data

Available shortly from EMPIAR-10677.

## Running the Code

Example data is provided in the membrane_curvature directory. To run the script:

```
python surface_curvature.py data_config.yaml
```

All parameters needed for measurement, including filenames, are encapsulated in the .yaml config file. The yaml file contains detailed notes describing each parameter.
