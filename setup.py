import contextlib
import os

from setuptools import find_packages, setup

with open(os.devnull, "w") as nullfile:
    with contextlib.redirect_stderr(nullfile):
        setup(
            name="scentinel",
            version="0.2.1",
            packages=find_packages(),
            install_requires=[
                "scanpy==1.9.3",
                "pandas==1.5.3",
                "numpy==1.22.1",
                "scipy==1.7.3",
                "matplotlib==3.7.1",
                "seaborn==0.12.2",
                "scikit-learn",
                "requests==2.31.0",
                "psutil==5.9.5",
                "mygene==3.2.2",
                "gseapy==1.0.5",
                "pymc3==3.11.5",
                "scikit-optimize",
            ],
        )
