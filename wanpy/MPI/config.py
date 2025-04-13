# Copyright (C) 2020 Jin Cao
#
# This file is distributed as part of the wanpy code and
# under the terms of the GNU General Public License. See the
# file LICENSE in the root directory of the wanpy
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The wanpy code is hosted on GitHub:
#
# https://github.com/jincao2013/wanpy

__date__ = "Apr. 4, 2025"

import time
import tomllib
import numpy as np
from wanpy import __version__, __author__

class Config:
    def __init__(self, MPI):
        # MPI-related variable
        self.comm = MPI.COMM_WORLD
        self.MPI_rank = self.comm.Get_rank()
        self.MPI_ncore = self.comm.Get_size()
        self.MPI_main = not self.MPI_rank

        self.job = None

        # System
        self.htb_dir = None
        self.htb_fname = None
        self.pt_symmetric = None

        # Tensor
        self.tensor_indices = None
        self.ntensor = None

        # DOS parameters
        self.ne = None
        self.emin = None
        self.emax = None
        self.ee = None
        self.omega = None
        self.ewidth = None

        # Fermi surface parameters
        self.gateMIN = None
        self.gateMAX = None
        self.ngate = None
        self.gate = None

        # Thermal properties
        self.temperature = None
        self.ntemperature = None
        self.tau = None

        # HTB properties
        self.tmin_h = None
        self.tmin_r = None
        self.open_boundary = None
        self.istb = None
        self.use_wcc = None
        self.atomic_wcc = None

        # Magnetic properties
        self.mag_sublatt_def = None
        self.nsublatt = None

        # Brillouin Zone properties
        self.nkmesh = None
        self.kcube = None
        self.kmesh_shift = None
        self.random_k = None
        self.centersym = None

        # Computed properties
        self.ewidth_imag = None

        # io
        self.iterprint = None

        '''
          * to add
        '''

        '''
          * creat when running
        '''
        self.timestamp = [0, 0]
        self.nk = None

    def load_config(self, fname='config.toml'):
        """Load parameters from a TOML configuration file."""
        with open(fname, "rb") as f:
            toml_dict = tomllib.load(f)

        # Job
        self.job = toml_dict.get("job")

        # System
        system = toml_dict.get("system", {})
        self.htb_dir = system.get("htb_dir")
        self.htb_fname = system.get("htb_fname")
        self.pt_symmetric = system.get("pt_symmetric")

        # Tensor
        tensor = toml_dict.get("tensor", {})
        self.tensor_indices = [tuple(t) for t in tensor.get("tensor_indices", [])]
        if self.tensor_indices is not None:
            self.ntensor = len(self.tensor_indices)

        # DOS
        dostags = toml_dict.get("dostags", {})
        self.ne = dostags.get("ne")
        self.emin = dostags.get("emin")
        self.emax = dostags.get("emax")
        if self.ne is not None:
            self.ee = np.linspace(self.emin, self.emax, self.ne)
        self.omega = dostags.get("omega")
        self.ewidth = dostags.get("ewidth")

        # Fermi Surface
        fermi_surface = toml_dict.get("fermi_surface", {})
        self.gateMIN = fermi_surface.get("gateMIN")
        self.gateMAX = fermi_surface.get("gateMAX")
        self.ngate = fermi_surface.get("ngate")
        if self.ngate is not None:
            self.gate = np.linspace(self.gateMIN, self.gateMAX, self.ngate)

        # Thermal
        thermal = toml_dict.get("thermal", {})
        self.temperature = np.array(thermal.get("temperature"), dtype="float64")
        if self.temperature is not None:
            self.ntemperature = self.temperature.shape[0]
        self.tau = thermal.get("tau")

        # HTB
        htb = toml_dict.get("htb", {})
        self.tmin_h = htb.get("tmin_h", -1e-6)
        self.tmin_r = htb.get("tmin_r", -1e-6)
        self.open_boundary = htb.get("open_boundary", -1)
        self.istb = htb.get("istb", True)
        self.use_wcc = htb.get("use_wcc", True)
        self.atomic_wcc = htb.get("atomic_wcc", True)

        # Magnetic
        magnetic = toml_dict.get("magnetic", {})
        self.mag_sublatt_def = np.array(magnetic.get("mag_sublatt_def", []))
        if self.mag_sublatt_def is not None:
            self.nsublatt = self.mag_sublatt_def.shape[0]

        # BZ
        BZ = toml_dict.get("BZ", {})
        self.nkmesh = np.array(BZ.get("nkmesh"), dtype="int64")
        self.kcube = np.array(BZ.get("kcube", np.identity(3).tolist()), dtype="float64")
        self.kmesh_shift = np.array(BZ.get("kmesh_shift", [0, 0, 0]), dtype="float64")
        self.random_k = BZ.get("random_k", False)
        self.centersym = BZ.get("centersym", False)

        # Initialize k-mesh after loading configuration
        # self.initialize_kmesh()

        # Artificial
        artificial = toml_dict.get("artificial", {})
        self.ewidth_imag = artificial.get("ewidth_imag")

        # output
        output = toml_dict.get("output", {})
        self.iterprint = int(output.get("iterprint", 200))

    def print_config(self):
        if not self.MPI_main: return

        print('Running on   {} total cores'.format(self.MPI_ncore))
        print('WanPy version: {}'.format(__version__))
        print('Author: {}'.format(__author__))
        print()

        print("\n=== CONFIGURATION ===")

        # Job
        print("\n[Job]")
        print(f"  job: {self.job}")

        # System
        print("\n[System]")
        print(f"  htb_dir: {self.htb_dir}")
        print(f"  htb_fname: {self.htb_fname}")
        print(f"  pt_symmetric: {self.pt_symmetric}")

        # Tensor
        print("\n[Tensor]")
        print(f"  tensor_indices: {self.tensor_indices}")
        print(f"  ntensor: {self.ntensor}")

        # DOS
        print("\n[DOS]")
        print(f"  ne: {self.ne}")
        print(f"  emin: {self.emin}")
        print(f"  emax: {self.emax}")
        print(f"  ee: \n{self.ee}")
        print(f"  omega: {self.omega}")
        print(f"  ewidth: {self.ewidth}")

        # Fermi Surface
        print("\n[Fermi Surface]")
        print(f"  gateMIN: {self.gateMIN}")
        print(f"  gateMAX: {self.gateMAX}")
        print(f"  ngate: {self.ngate}")
        print(f"  gate: \n{self.gate}")

        # Thermal
        print("\n[Thermal]")
        print(f"  temperature: {self.temperature}")
        print(f"  ntemperature: {self.ntemperature}")
        print(f"  tau: {self.tau}")

        # HTB Properties
        print("\n[HTB]")
        print(f"  tmin_h: {self.tmin_h}")
        print(f"  tmin_r: {self.tmin_r}")
        print(f"  open_boundary: {self.open_boundary}")
        print(f"  istb: {self.istb}")
        print(f"  use_wcc: {self.use_wcc}")
        print(f"  atomic_wcc: {self.atomic_wcc}")

        # Magnetic Properties
        print("\n[Magnetic]")
        print(f"  mag_sublatt_def: \n{self.mag_sublatt_def}")
        print(f"  nsublatt: \n{self.nsublatt}")

        # Brillouin Zone Properties
        print("\n[Brillouin Zone]")
        print(f"  nkmesh: {self.nkmesh}")
        print(f"  kcube: \n{self.kcube}")
        print(f"  kmesh_shift: {self.kmesh_shift}")
        print(f"  Random K: {self.random_k}")
        print(f"  centersym: {self.centersym}")

        # Artificial
        print("\n[Artificial]")
        print(f"  ewidth_imag: {self.ewidth_imag}")

        # output
        print("\n[output]")
        print(f"  iterprint: {self.iterprint}")

        print("\n=====================")

    def start_timer(self):
        if self.MPI_main:
            self.timestamp[0] = time.time()
            print('{} job start at {}.'.format(self.job, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def end_timer(self):
        if self.MPI_main:
            self.timestamp[1] = time.time()
            duration = self.timestamp[1] - self.timestamp[0]
            print('Time consuming in total {:.3f}s ({:.3f}h)'.format(duration, duration/3600))
            print('Time per k-point per core: {:.2f}ms (nk = {})'.format(1000 * self.MPI_ncore * duration / self.nk, self.nk))

    def dict_serializable(self):
        return {
            key: val for key, val in self.__dict__.items()
            if key not in ['comm']
        }

# if __name__ == "__main__":
#     # Example usage:
#
#     from mpi4py import MPI
#     config = Config(MPI)
#     config.load_config("config.toml")
#     config.print_config()
#     # config.__dict__