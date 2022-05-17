#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )

install_req = [
    "numpy>=1.7",
]
tests_req = [
    "pytest>=7.1.1",
    "pytest-rng>=1.0.0",
    "tensorflow>=2.2.0",
]

setup(
    name="np-conv2d",
    version="1.0.0",
    author="Mengye Ren",
    author_email="renmengye@gmail.com",
    py_modules=["conv2d"],
    url="https://github.com/renmengye/np-conv2d",
    license="MIT license",
    description="2D Convolution using NumPy",
    install_requires=install_req,
    extras_require={
        "tests": tests_req,
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)
