# WanPy: A Wannier Tight-Binding Model Based Numerical Package

## Requirements
WanPy requires the following packages to be installed:
- numpy
- scipy
- pandas
- matplotlib
- h5py
- mpi4py

## Installation
1. To install WanPy, create a Conda environment with the required packages:
```bash
conda create -n wanpy python=3.9 numpy scipy pandas matplotlib h5py mpi4py
````

2. Setup environment variables: 

For personal computer:
```bash
export PYTHONPATH="path_of_wanpy:$PYTHONPATH"
export PATH="path_of_wanpy/scripts:$PATH"
export WANPY_ROOT_DIR="your_work_dir"
export PYGUI="True"
```

For clusters:
```bash
export PYTHONPATH="path_of_wanpy:$PYTHONPATH"
export PATH="path_of_wanpy/scripts:$PATH"
```

## License
WanPy is released under the GNU General Public License. See LICENSE for details.