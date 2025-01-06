from setuptools import setup, find_packages, find_namespace_packages

requirements = {"core": [], "optional": {}}
# Read dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    sections = f.read().split("# Optional dependencies")  # Split the content into sections

# Process core dependencies
requirements["core"] = [line for line in sections[0].strip().splitlines() if "==" in line]

# Process optional dependencies
if len(sections) > 1:
    requirements["optional"] = {line.split("==")[0]:line for line in sections[1].strip().splitlines() if "==" in line}
    requirements["optional"]["all"] = list(set(dep for deps in requirements["optional"].values() for dep in deps))

setup(
    name="langswarm-memory",
    version="0.0.1",
    description = (
        "LangSwarm-Memory: A versatile memory management framework for multi-agent systems, "
        "supporting advanced retrieval, reranking workflows, and domain-specific templates "
        "for autonomous AI solutions."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aekdahl/langswarm-memory",
    author="Alexander Ekdahl",
    author_email="alexander.ekdahl@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_namespace_packages(include=["langswarm.*"]),
    python_requires=">=3.8",
    install_requires=requirements["core"],
    extras_require=requirements["optional"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            # If your package includes CLI tools, specify them here.
            # e.g., "langswarm=core.cli:main",
        ],
    },
)
