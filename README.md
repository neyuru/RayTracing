# RayTracing
A Python script designed to simulate the propagation of rays through heterogeneous media. It models the interaction of rays with different media, allowing the analysis of various scenarios and phenomena according to the principle of least time.

# Ray Tracing Simulation in Heterogeneous and Isotropic Media

This Python script simulates ray propagation in heterogeneous and isotropic media by solving the equations of motion, obtained by Fermat's principle, using a shooting method. The simulations and results of this project have been used in several research articles, including "On ray tracing for sharp changing media" published by the Journal of the Acoustical Society of America.

## Overview

The script allows users to:
- Select from predefined scenarios and methods of solution to the equations of motion.
- Choose to display results or benchmark the selected combination of scenarios/methods by measuring completion times.
- Find an appropriate `DELTA_S` which is a crucial parameter related to the step size that a ray would advance in each iteration.

## Features

- Multiple Scenarios: Includes various scenarios like `interface`, `fisheye`, `isotropy`.
- Different Solution Methods: Users can choose different methods to solve the equations of motion.
- Benchmarking: Allows users to measure completion times for the selected combination of scenarios/methods.
- Visualization: Provides plots of scenarios and ray parameters.

## Usage

This script was successfully tested on Python 3.11.5 but should work on other versions as well. It requires the following libraries (with tested versions, but other versions may also work): 
- numpy (1.26.0)
- scipy (1.11.2)
- matplotlib (3.8.0)

Additionally, a working installation of Latex is also required for rendering text in plots. To run the script, navigate to the project directory and execute:

```sh
python RT_bench.py
```

The user will be prompted to select an algorithm option, choose whether to find an appropriate `DELTA_S`, and decide whether to proceed with the benchmarking process.

## Related Articles

- "On ray tracing for sharp changing media" - Journal of the Acoustical Society of America.
- Another article is under peer review at the time of this writing.

## License

This project is open source and available under the [MIT License](https://opensource.org/license/mit/).

## Author

Jorge A. Ramos O.
jorge.ro at saltillo.tecnm.mx

## Citation

If you use this code in your research, please cite it as follows:

```bibtex
@software{jorge_alberto_ramos_oliveira_2023_8387038,
  author       = {Jorge Alberto Ramos Oliveira},
  title        = {Ray Propagation Simulation in Heterogeneous and Isotropic Media},
  month        = sep,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.2.0},
  doi          = {10.5281/zenodo.8387038},
  url          = {https://doi.org/10.5281/zenodo.8387038}
}
```
