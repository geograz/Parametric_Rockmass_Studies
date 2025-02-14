# Parametric_Rockmass_Studies
Quantifying rock mass structure for engineering purposes has always been a central topic of rock engineering. Some of the used methods today (2025) are, however, decades old and could benefit from modern computational tools. "Parametric Rock Mass Studies" comprises a set of investigations that review and revise methods of rock mass structure characterization.

The theoretical approach is heavily based on discrete discontinuity models and a first benchmark dataset of 5000 discontinuity models - called **PDD1** - was generated in the first study by Erharter (2024).

This repository contains the code and generated data that is associated with the studies. All samples of **PDD1** can be found in the folder `discontinuities`.

- The first study (Erharter 2024) was focused on rock mass discontinuity network characterizations and was mainly working with the mesh models of the discontinuities of **PDD1**.
- The second study (Erharter and Elmo 2025) builds directly on the first and is focusing on computationally quantifying rock mass complexity as a whole and tries to answer the question of whether rock mass can be treated as a continuum or a discontinuum.
- Future studies are planned and will investigate other advanced aspects of rock mass characterization.


## Publications
Methods and discussions of the results of the parametric rock mass studies can be found in the publications:
- ERHARTER, Georg H. (2024) Rock Mass Structure Characterization Considering Finite and Folded Discontinuities: A Parametric Study. Rock Mechanics and Rock Engineering. https://doi.org/10.1007/s00603-024-03787-9
- A second manuscript with the title "Is Complexity the Answer to the Continuum vs. Discontinuum Question in Rock Engineering?" by Georg Erharter and Davide Elmo was submitted to the journal Rock Mechanics and Rock Engineering in February 2025 and is currently under review.


## Repository structure
```
Parametric_Rockmass_Studies
├── combinations					- folder that contains all the 5000 individual samples of PDD1.
├── output							- folder that contains computational results of analyses of the samples
│   ├── graphics					- folder that contains visualizations of the result analyses
│   ├── df_samples.csv				- log file that indicates the processing state of raster samples of different resolutions
│   ├── parameters.xlsx			- OUTDATED excel files that contains descriptions for the parameters collected as part of PDD1 generation in Erharter (2024)
│   ├── PDD1_0.xlsx				- result excel file containing input parameters for PDD1 generation from Erharter (2024) and compiled Grasshopper analyses
│   ├── PDD1_1.xlsx				- result excel file extended from PDD1_0.xlsx that also contains basic voxel data parameters
│   ├── PDD1_2.xlsx				- final result excel file extended from PDD1_1.xlsx that also contains advanced complexity analyses parameters
├── src							- folder with Python scripts
│   ├── A_compiler.py				- Script that compiles the recorded data from samples of the discrete discontinuity networks and creates excel files PDD1_0.xlsx and PDD1_1.xlsx for further processing.
│   ├── B_analyzer.py				- Script that processes the compiled records of the discrete discontinuity dataset, computes additional and complexity parameters. Output: PDD1_2.xlsx
│   ├── C_rasterizer.py			- Script that loads meshes and computes rasters at different resolution from them for further analyses. Does not include raster analyses - only generation.
│   ├── D_Voxel_export.py			- Script that saves a mesh of a rastered sample. For visualization purposes only.
│   ├── E_plotter.py				- Script that produces plots and visualizations from the rock mass analyses.
│   ├── F_MeshChecker.py			- Script that performs different checks of the generated meshes.
│   ├── X_library.py				- Script that contains a custom library with different classes of functions for math, general use (utilities), plotting and computation of parameters.
├── .gitignore
├── environment.yaml				- dependency file to use with conda
├── Grasshopper.zip				- zipped folder containing the Grasshopper script that was used in Erharter (2024) to generate PDD1 and also the direct Grasshopper output files that specify sample input and measurements in Grasshopper
├── LICENSE						- file with specifications of the applied MIT License
├── README.md
```

## Requirements

The environment is set up using `conda`.

To do this create an environment called `Jv` using `environment.yaml` with the help of `conda`.
```bash
conda env create --file environment.yaml
```

Activate the new environment with:

```bash
conda activate Jv
```

### contact
georg.erharter@ngi.no
