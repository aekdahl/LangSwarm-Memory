from setuptools import setup, find_packages

setup(
    name="memswarm",
    version="1.0.0",
    description="A hybrid shared memory system with scoped access for multi-agent systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/memswarm",
    author="Your Name",
    author_email="your.email@example.com",
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
