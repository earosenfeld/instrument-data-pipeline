from setuptools import setup, find_packages

setup(
    name="instrument-data-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "sqlalchemy>=1.4.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
    ],
) 