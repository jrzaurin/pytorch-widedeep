#!/usr/bin/env python3
# flake8: noqa
import os
import setuptools

pwd = os.path.dirname(__file__)

dev_status = {
    "0.1": "Development Status :: 1 - Planning",  # v0.1 - skeleton
    "0.2": "Development Status :: 2 - Pre-Alpha",  # v0.2 - some basic functionality
    "0.3": "Development Status :: 3 - Alpha",  # v0.3 - most functionality
    "0.4": "Development Status :: 4 - Beta",  # v0.4 - most functionality + doc
    "1.0": "Development Status :: 5 - Production/Stable",  # v1.0 - most functionality + doc + test  # noqa
    "2.0": "Development Status :: 6 - Mature",  # v2.0 - new functionality?
}


with open(os.path.join(pwd, "VERSION")) as f:
    version = f.read().strip()
    assert len(version.split(".")) == 3, "bad version spec"
    majorminor = version.rsplit(".", 1)[0]


# with open("README.md", "r") as f:
#     long_description = f.read()

long_description ="""
pytorch-widedeep: Modular and flexible package to combine tabular data with text and
images using Wide and Deep models in Pytorch

For an introduction to the package and a quick start, go to:

    https://github.com/jrzaurin/pytorch-widedeep

For a temporal documentation, go to:

    https://github.com/jrzaurin/pytorch-widedeep/tree/master/docs

You can find the source code at:

    https://github.com/jrzaurin/pytorch-widedeep/tree/master/pytorch_widedeep

"""

# main setup kw args
setup_kwargs = {
    "name": "pytorch-widedeep",
    "version": version,
    "description": "Combine tabular data with text and images using Wide and Deep models in Pytorch",
    # "long_description_content_type": 'text/markdown',
    "long_description": long_description,
    "author": "Javier Rodriguez Zaurin",
    "author_email": "jrzaurin@gmail.com",
    "url": "https://github.com/jrzaurin/pytorch-widedeep",
    "license": "MIT",
    "install_requires": [
        "pytest",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "gensim",
        "spacy",
        "opencv-contrib-python",
        "imutils",
        "tqdm",
        "torch",
        "torchvision",
    ],
    "classifiers": [
        dev_status[majorminor],
        "Environment :: Other Environment",
        "Framework :: Jupyter",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    "zip_safe": True,
    "packages": setuptools.find_packages(exclude=["test_*.py"]),
}


if __name__ == "__main__":
    setuptools.setup(**setup_kwargs)
