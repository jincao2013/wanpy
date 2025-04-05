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

import os

def main():
    path_scripts = os.path.dirname(os.path.abspath(__file__))
    path_site_packages = os.path.dirname(os.path.dirname(path_scripts))
    path_projects = os.path.join(path_site_packages, "wanpyProjects")  # Source directory
    dest = os.path.expanduser("~/opt/wanpyProjects")  # Destination directory

    if os.path.exists(dest) or os.path.islink(dest):
        raise Exception(f"file {dest} exist. ")
        # os.remove(dest)

    os.symlink(path_projects, dest)
    print(f"Symlink created: {dest} -> {path_projects}")

if __name__ == "__main__":
    main()
