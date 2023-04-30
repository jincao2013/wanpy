# WanPy: A Wannier Tight-Binding Model Based Numerical Package

## Requirements
WanPy requires the following packages to be installed:
- numpy
- scipy
- pandas
- sympy
- matplotlib
- h5py
- mpi4py

## Installation
1. To install WanPy, create a Conda environment with the required packages:
```bash
$ conda create -n wanpy python=3.7 numpy scipy pandas sympy matplotlib h5py mpi4py
````

2. Uncompress wanpy package in personal computer or clusters: 

```bash
$ tar zxvf wanpy.tar.gz
```
or
```bash
$ git clone https://github.com/jincao2013/wanpy.git
```

3. Setup environment variables: 

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

## Get start
```bash
# show usage
$ wanpy -h

# gather Wannier tb data
$ wanpy htb

# get Wannier tb model
$ wanpy wannier

# compare Wannier and VASP band structure 
$ wpyplotband

# check job status
$ statistic_cores ...
```

## License
WanPy is released under the GNU General Public License. See LICENSE for details.
