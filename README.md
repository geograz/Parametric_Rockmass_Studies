# Parametric_Rockmass_Studies
Quantifying rock mass structure for engineering purposes has always been a central topic of rock engineering. Some of the used methods today (2025) are, however, decades old and could benefit from modern computational tools. "Parametric Rock Mass Studies" comprises a set of investigations that review and revise methods of rock mass structure characterization.

The theoretical approach is heavily based on discrete discontinuity models and a first benchmark dataset of 5000 discontinuity models - called **PDD1** - was generated in the first study by Erharter (2024).

This repository contains the code and generated data that is associated with the studies. All samples of **PDD1** can be found in the folder `discontinuities`.

- The first study (Erharter 2024) was focused on rock mass discontinuity network characterizations and was mainly working with the mesh models of the discontinuities of **PDD1**.
- The second study (Erharter and Elmo 2025) builds directly on the first and is focusing on computationally quantifying rock mass complexity as a whole and tries to answer the question of whether rock mass can be treated as a continuum or a discontinuum.
- Future studies are planned and will investigate other advanced aspects of rock mass characterization.


## Publications
The following publications have been produced as part of the Parametric Rock Mass Studies:
- ERHARTER Georg H. (2024) Rock Mass Structure Characterization Considering Finite and Folded Discontinuities: A Parametric Study. Rock Mechanics and Rock Engineering. https://doi.org/10.1007/s00603-024-03787-9
- ERHARTER Georg H. and ELMO Davide (2025): Is Complexity the Answer to the Continuum vs. Discontinuum Question in Rock Engineering?. Rock Mechanics and Rock Engineering. STATUS MANUSCRIPT ACCEPTED


## Supplementary data
Due to size limitations on Github, large data has to be moved to other repositories. Exemplary sample data from PDD1 sample with ID 151763271961 can be found in the folder 'sample_data'. Two external datasets are currently connected to the repository:

- The original **Parametric Discontinuity Dataset 1 (PDD1)**: the dataset was created as part of Erharter (2024). It consists of 5000 meshes that comprise synthetic rock mass models in the form of 10 x 10 x 10 m cubes with discontinuities. It can be found on *Zenodo* under this link: https://doi.org/10.5281/zenodo.15835130
- The **raster models of PDD1** that were created as part of Erharter and Elmo (2025). All 5000 models of PDD1 were rastered at resolutions of 0.25, 0.2, 0.15, 0.1 and 0.05 m. Total = 25 000 rasters. The dataset can be found on *Zenodo* under this link: https://doi.org/10.5281/zenodo.15570244

Full Data references:
- Erharter, G. (2025). Parametric Discontinuity Dataset 1 (PDD1) (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15835130
- Erharter, G. (2025). Raster models for paper "Is Complexity the Answer to the Continuum vs. Discontinuum Question in Rock Engineering?" by G. Erharter and D. Elmo (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15570244


## Repository structure
```
Parametric_Rockmass_Studies
├── output			- folder that contains computational results of analyses of the samples
│   ├── graphics			- folder that contains visualizations of the result analyses
│   ├── df_samples.csv			- log file that indicates the processing state of raster samples of different resolutions
│   ├── PDD1_0.xlsx			- result excel file containing input parameters for PDD1 generation from Erharter (2024) and compiled Grasshopper analyses
│   ├── PDD1_1.xlsx			- result excel file extended from PDD1_0.xlsx that also contains basic voxel data parameters
│   ├── PDD1_2.xlsx			- final result excel file extended from PDD1_1.xlsx that also contains advanced complexity analyses parameters
├── sample_data			- folder with exemplary synthetic rock mass models as meshes
├── src			- folder with Python scripts
│   ├── A_compiler.py			- Script that compiles the recorded data from samples of the discrete discontinuity networks and creates excel files PDD1_0.xlsx and PDD1_1.xlsx for further processing.
│   ├── B_analyzer.py			- Script that processes the compiled records of the discrete discontinuity dataset, computes additional and complexity parameters. Output: PDD1_2.xlsx
│   ├── C_rasterizer.py			- Script that loads meshes and computes rasters at different resolution from them for further analyses. Does not include raster analyses - only generation.
│   ├── D_Voxel_export.py			- Script that saves a mesh of a rastered sample. For visualization purposes only.
│   ├── E_plotter.py			- Script that produces plots and visualizations from the rock mass analyses.
│   ├── F_MeshChecker.py			- Script that performs different checks of the generated meshes.
│   ├── G_animation.py			- Script that produces an animation from frames to visualize synthetic rock mass models and other 3D models.
│   ├── X_library.py			- Script that contains a custom library with different classes of functions for math, general use (utilities), plotting and computation of parameters.
├── .gitignore
├── environment.yaml			- dependency file to use with conda
├── Grasshopper.zip			- zipped folder containing the Grasshopper script that was used in Erharter (2024) to generate PDD1 and also the direct Grasshopper output files that specify sample input and measurements
├── LICENSE			- file with specifications of the applied MIT License
├── README.md
```


## Requirements

The environment is set up using `conda`.

To do this create an environment called `Jv` using `environment.yaml` with the help of `conda`.
```bash
conda env create --file environment.yaml
```


### contact
georg.erharter@ngi.no
