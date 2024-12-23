from setuptools import setup, find_packages

setup(
    name="memswarm",
    version="0.0.1",
    description="A hybrid shared memory system with scoped access for multi-agent systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aekdahl/memswarm",
    author="Alexander Ekdahl",
    author_email="alex@zubi.ai",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "redis",
        "chromadb",
        "google-cloud-storage",
        "sqlalchemy",
        "sentence-transformers",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
