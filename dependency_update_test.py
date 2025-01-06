import subprocess
import requests
import sys
from packaging.version import Version

def fetch_versions(package_name):
    """
    Fetch all available versions of a package from PyPI.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        all_versions = list(response.json()["releases"].keys())
        # Sort versions using `packaging.version.Version`
        all_versions.sort(key=Version)
        return all_versions
    except requests.RequestException as e:
        print(f"Error fetching versions for {package_name}: {e}")
        return []

def test_dependency_version(package, version):
    """
    Test if a specific version of a package can be installed.
    """
    try:
        print(f"Testing {package}=={version}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{package}=={version}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"{package}=={version} installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print(f"{package}=={version} failed.")
        return False

def find_oldest_compatible_version(package, versions):
    """
    Find the oldest compatible version of a package by testing all versions.
    """
    compatible_version = None
    for version in reversed(versions):  # Test oldest versions first
        if test_dependency_version(package, version):
            compatible_version = version
            break
    return compatible_version

def get_supported_python_versions():
    """
    Extract the list of supported Python versions from the requirements.txt file.
    """
    supported_versions = []
    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                if line.startswith("# Supported versions of Python:"):
                    supported_versions = line.strip().split(":")[1].strip().split(", ")
                    break
    except FileNotFoundError:
        pass
    return supported_versions

def update_requirements_with_python_versions(dependency_versions, python_version, success, extras_require):
    """
    Update the requirements.txt file with the latest compatible versions
    and maintain only supported Python versions.
    """
    supported_versions = set(get_supported_python_versions())

    if success:
        supported_versions.add(python_version)  # Add the Python version if it succeeded
    else:
        supported_versions.discard(python_version)  # Remove the version if it failed

    supported_versions = sorted(supported_versions)

    with open("requirements.txt", "w") as f:
        f.write(f"# Supported versions of Python: {', '.join(supported_versions)}\n")
        f.write("# Automatically updated by dependency_update_test.py\n\n")

        f.write("# Core dependencies\n")
        for package, compatible_version in dependency_versions["core"].items():
            f.write(f"{package}=={compatible_version}\n")

        for group, packages in extras_require.items():
            f.write(f"\n# Optional dependencies: {group}\n")
            for package, compatible_version in packages.items():
                f.write(f"{package}=={compatible_version}\n")
    print("requirements.txt updated successfully with Python version support and extras_require.")

def parse_requirements_file():
    """
    Parse the requirements.txt file into core and optional dependencies.
    """
    core_dependencies = []
    optional_dependencies = {}
    current_group = None

    try:
        with open("requirements.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# Optional dependencies: "):
                    current_group = line.split(": ")[1]
                    optional_dependencies[current_group] = {}
                elif "==" in line:
                    package, version = line.split("==")
                    if current_group:
                        optional_dependencies[current_group][package] = version
                    else:
                        core_dependencies.append(package)
    except FileNotFoundError:
        print("requirements.txt not found.")
        sys.exit(1)

    return core_dependencies, optional_dependencies

def main(python_version):
    core_dependencies, optional_dependencies = parse_requirements_file()
    latest_versions = {"core": {}, "optional": {}}  # Store compatible versions

    for package in core_dependencies:
        print(f"\nFetching versions for core dependency {package}...")
        versions = fetch_versions(package)
        if not versions:
            print(f"No versions found for {package}. Skipping...")
            continue

        compatible_version = find_oldest_compatible_version(package, versions)
        if compatible_version:
            latest_versions["core"][package] = compatible_version

    for group, packages in optional_dependencies.items():
        latest_versions["optional"].setdefault(group, {})
        for package in packages:
            print(f"\nFetching versions for optional dependency {package} (group: {group})...")
            versions = fetch_versions(package)
            if not versions:
                print(f"No versions found for {package}. Skipping...")
                continue

            compatible_version = find_oldest_compatible_version(package, versions)
            if compatible_version:
                latest_versions["optional"][group][package] = compatible_version

    extras_require = {
        group: {
            package: version
            for package, version in packages.items()
        }
        for group, packages in latest_versions["optional"].items()
    }

    update_requirements_with_python_versions(
        latest_versions["core"], python_version, True, extras_require
    )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dependency_update_test.py <python_version>")
        sys.exit(1)
    main(sys.argv[1])
