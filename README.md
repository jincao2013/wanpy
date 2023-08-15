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

For debug in personal computer:
```bash
export PYTHONPATH="path_of_wanpy:$PYTHONPATH"
export PATH="path_of_wanpy/scripts:$PATH"
export WANPY_ROOT_DIR="your_work_dir"
export PYGUI="True"
```

For deployment in clusters:
```bash
export PYTHONPATH="path_of_wanpy:$PYTHONPATH"
export PATH="path_of_wanpy/scripts:$PATH"
```

## Get start
```bash
# show usage
$ wanpy -h

# collect Wannier TB data to a single .h5 file
$ wanpy htb

# Collect Wannier TB data to a single .h5 file
$ wanpy wannier

# compare Wannier and VASP band structure. support HSE type calculation. 
$ wpyplotband

# check job status
$ statistic_cores ...
```

## Collect Wannier TB data

WanPy can collect necessary Wannier TB data into a single `.h5` file as the input for further calculations. This can be done by: 



**Method 1**: from `wanpy htb`, this will need at least `POSCAR, .wout, _hr.dat `. The files `.nnkp`,  `_r.dat`, `_wsvec.dat`, and `_spin.dat` are optional. 

```bash
# start to collect Wannier TB data 
$ wanpy htb [options]

# For detail of options, see
$ wanpy htb -h

```



**Method 2**: from `wanpy wannier`, this will need `.nnkp, .wout, .chk, .eig`, and `WAVECAR` if  `--spn`. The `nnkp` file can be obtained by: 

```bash
$ wannier90.x -pp wannier90.win 
```

To start collecting dat:

```bash
# start to collect Wannier TB data 
$ wanpy wannier [options]

# For detail of options, see
$ wanpy wannier -h

```



**Symmetric Wannier tight-binding models (SWTB)**

Set symmetry operations that used to symmetrize the model in `symmetry.in`, 

```bash
ngridR = 14 14 14  # use a larger value than the original Wannier TB model
&symmops
# TR  det  alpha  nx  ny  nz  taux  tauy  tauz
  1   1    0      0   0   1   0     0     0   # e
  1   1    180    0   0   1   0.5   0.5   0   # c2z
# anti unitary
 -1   1    0      0   0   1   0.5   0.5   0   # T
 -1   1    180    0   0   1   0     0     0   # Tc2z
/
```

a template file can also be produced by executing `wanpy --temp`. Then start to get SWTB with an additional tag `--symmetry`

```bash
$ wanpy wannier --symmetry [options]
```



## License

WanPy is released under the GNU General Public License. See LICENSE for details.
