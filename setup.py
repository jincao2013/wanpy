from setuptools import setup
from wanpy import __version__, __author__, __email__

packages_wanpy = [
    'wanpy',
    'wanpy.core',
    'wanpy.interface',
    'wanpy.model',
    'wanpy.MPI',
    'wanpy.response',
    'projects',
]

scripts_wanpy = [
    'scripts/wanpy',
    # 'scripts/wpy_get_poscar_from_wout.py',
    # 'scripts/wpy_set_fermi_eq_zero.py',
    'scripts/wpyplotband',
    'scripts/statistic_cores',
    'scripts/findmsg',
]

# to release package: python setup.py sdist
setup(
    name='wanpy',
    version=__version__,
    # include_package_data=True,
    packages=packages_wanpy,
    # packages=setuptools.find_packages(),
    scripts=scripts_wanpy,
    url='https://github.com/jincao2013/wanpy',
    license='GPL v3',
    author=__author__,
    author_email=__email__,
    description='This is the wanpy module.',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'sympy',
        'matplotlib',
        'pandas',
        'mpi4py',
        'h5py',
        'fortio',
        'spglib>=2.3.0'
    ],
)
