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

extras = {}
extras["test"] = ["pytest"]
extras["docs"] = [
    "sphinx",
    "sphinx_rtd_theme",
    "recommonmark",
    "sphinx-markdown-tables",
    "sphinx-copybutton",
    "sphinx-autodoc-typehints",
]
extras["quality"] = [
    "black",
    "isort @ git+git://github.com/timothycrosley/isort.git@e63ae06ec7d70b06df9e528357650281a3d3ec22#egg=isort",
    "flake8",
]

# main setup kw args
setup_kwargs = {
    "name": "pytorch-widedeep",
    "version": version,
    "description": "Combine tabular data with text and images using Wide and Deep models in Pytorch",
    "long_description": open("pypi_README.md", "r", encoding="utf-8").read(),
    "long_description_content_type": "text/markdown",
    # "long_description": long_description,
    "author": "Javier Rodriguez Zaurin",
    "author_email": "jrzaurin@gmail.com",
    "url": "https://github.com/jrzaurin/pytorch-widedeep",
    "license": "MIT",
    "install_requires": [
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
    "extra_requires": extras,
    "python_requires": ">=3.6.0",
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
