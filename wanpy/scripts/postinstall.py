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
