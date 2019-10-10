# From https://github.com/KristianHolsheimer/keras-gym
#!/usr/bin/env python3
# flake8: noqa
import os
import setuptools

pwd = os.path.dirname(__file__)

with open(os.path.join(pwd, 'version.txt')) as f:
    version = f.read().strip()
    assert len(version.split('.')) == 3, "bad version spec"
    majorminor = version.rsplit('.', 1)[0]


dev_status = {
    '0.1': 'Development Status :: 1 - Planning',          # v0.1 - skeleton
    '0.2': 'Development Status :: 2 - Pre-Alpha',         # v0.2 - some basic functionality
    '0.3': 'Development Status :: 3 - Alpha',             # v0.3 - most functionality
    '0.4': 'Development Status :: 4 - Beta',              # v0.4 - most functionality + doc
    '1.0': 'Development Status :: 5 - Production/Stable', # v1.0 - most functionality + doc + test  # noqa
    '2.0': 'Development Status :: 6 - Mature',            # v2.0 - new functionality?
}


long_description = """
pytorch-widedeep: Easy-to-use modular Wide and Deep learning frame to
combine tabular, images and text datasets
"""

# main setup kw args
setup_kwargs = {
    'name': 'pytorch-widedeep',
    'version': version,
    'description': "Easy-to-use modular Wide and Deep learning frame in Pytorch",
    'long_description': long_description,
    'author': 'Javier Rodriguez Zaurin',
    'author_email': 'jrzaurin@gmail.com',
    'url': '',
    'license': 'MIT',
    'install_requires': [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "gensim",
        "imutils",
        "torch",
        "torchvision",
        "fastai",
        # "opencv-python",
        "tqdm"],
    'classifiers': [
        dev_status[majorminor],
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(exclude=['test_*.py']),
}


if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)