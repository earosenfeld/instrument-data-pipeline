from setuptools import setup, find_packages

setup(
    name="instrument-data-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
) 