# Parametric_Rockmass_Studies
Repository for the code to do parametric studies on the structure of rockmass.

## methods
Samples of discrete discontinuity networks are created with the visual programming language "Grasshopper" in the computer aided design software Rhino 3D (https://www.rhino3d.com/). "Virtual measurements" are then taken on these samples and used for further processing and investigation of rock mass parameters.

## publications
- A first manuscript about this study was submitted to the journal Rock Mechanics and Rock Engineering and is currently under review. Erharter, G.H.: "Rock mass structure characterization considering finite and folded discontinuities"

## repository structure
```
Parametric_Rockmass_Studies
├── src
│   ├── 00_Cluster.ghcluster              - grasshopper (Rhino) cluster that is used in main grasshopper script
│   ├── 00_main.gh                        - main grasshopper script that genarates parametric rock mass samples
│   ├── A_compiler.py                     - Script that compiles the recorded data from samples of the discrete discontinuity networks and creates one excel file for further processing.
│   ├── B_analyzer.py                     - Script that processes the compiled records of the discrete discontinuity dataset, computes new parameters and creates figures to visualize the dataset.
│   ├── C_boxcounting.py                  - Script that performs voxel based boxcounting that is used to estimate each samples' fractal dimension.
│   ├── D_function_vis.py                 - Script that generates specific plots for publications.
│   ├── X_library.py                      - Script that contains a custom library with different classes of functions for math, plotting or general use (utilities).
├── .gitignore
├── LICENSE                               - file with specifications of the applied MIT License
├── environment.yaml                      - dependency file to use with conda
├── README.md
```

## Requirements

The environment is set up using `conda`.

To do this create an environment called `Jv` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`
```bash
conda env create --file environment.yaml
```

Activate the new environment with:

```bash
conda activate Jv
```

### contact
georg.erharter@ngi.no
