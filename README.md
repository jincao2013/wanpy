# WanPy: A Wannier Tight-Binding Model Based Numerical Package

## Requirements
WanPy requires the following packages to be installed:
- numpy
- scipy
- pandas
- sympy
- h5py
- mpi4py
- fortio
- spglib
- matplotlib (optional)
- phonopy (optional)

## Installation
1. To install WanPy, create a Conda environment with the required packages:
```bash
$ conda create -n wanpy python numpy scipy pandas sympy h5py mpi4py fortio spglib matplotlib
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

# Write _hr.dat and _r.dat (optional) from .h5 file
$ wanpy write_dat

# Twist the order of .amn (uudd or udud) 
$ wanpy twist_amn

# compare Wannier and VASP band structure. support HSE type calculation. 
$ wpyplotband

# check job status
$ statistic_cores ...
```

## Collect Wannier TB data

WanPy need to collect necessary Wannier TB data into a single `.h5` file as the input for further calculations. This can be done by: 

**Important notice:**

WanPy employs the uudd order of Wannier orbitals for its internal calculations, which is the default setting when using v1.2 of `wannier_setup`. If a higher version is used, one should utilize `wanpy twist_amn` to reorganize the .amn file into the uudd order and then proceed with the disentanglement process once more.

**Method 1**: from `wanpy wannier`, this will need `POSCAR .nnkp, .wout, .chk, .eig`, and `WAVECAR` if  `--spn`. The `nnkp` file can be obtained by: 

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



**Method 2**: from `wanpy htb`, this will need at least `POSCAR, .wout, _hr.dat `. The files `.nnkp`,  `_r.dat`, `_wsvec.dat`, and `_spin.dat` are optional. 

```bash
# start to collect Wannier TB data 
$ wanpy htb [options]

# For detail of options, see
$ wanpy htb -h

```



**Symmetric Wannier tight-binding models (SWTB)**

The SWTB can be obtained during collecting Wannier TB data. Set symmetry related parameters in `symmetry.in`, a template file can also be produced by executing `wanpy wannier --temp`: 

```bash
# Input file for building symmetric Wannier TB model

# Choose method, default is kspace
symmetric_method = rspace   # Options: {rspace, kspace}

# Choose between manually setting symmops or automatically detecting symmops from magmoms
parse_symmetry = man        # Options: {man, auto}

ngridR = 12 12 1            # used in kspace method, and use a slightly larger value than the original TB model 

# Parameters used when parse_symmetry = auto
symprec = 1e-5
&magmoms
0 0 1
0 0 -1
/

# Parameters used when parse_symmetry = man
&symmops
# TR  det  alpha  nx  ny  nz  taux  tauy  tauz
  0   1    0      0   0   1   0     0     0   # e
  0   1    180    0   0   1   0.5   0.5   0   # c2z
# Anti-unitary symmetry operations
  1   1    0      0   0   1   0.5   0.5   0   # T
  1   1    180    0   0   1   0     0     0   # Tc2z
/
```

Then start to get SWTB with an additional tag `--symmetry`

```bash
$ wanpy wannier --symmetry [options]
```



See [Comput. Phys. Commun. 270, 108153 (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0010465521002654) [(arXiv:2012.08871)](https://arxiv.org/abs/2105.09504) for detail. 



## License

WanPy is released under the GNU General Public License. See LICENSE for details.
