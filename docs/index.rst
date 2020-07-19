pytorch-widedeep
================

*A flexible package to combine tabular data with text and images using wide and deep models*

Below there is an introduction to the two architectures available in
``pytorch-widedeep``. If you prefer to learn about the utilities and
components go straight to the Documentation.

Documentation
-------------

.. toctree::
    :maxdepth: 1

    Installation <installation>
    Quick Start <quick_start>
    Utilities <utils/index>
    Preprocessing <preprocessing>
    Model Components <model_components>
    Wide and Deep Models <wide_deep/index>
    Examples <examples>


Introduction
------------
``pytorch-widedeep`` is based on Google's Wide and Deep Algorithm. Details of
the original algorithm can be found in this nice `tutorial
<https://www.tensorflow.org/tutorials/wide_and_deep>`_, and the `research
paper <https://arxiv.org/abs/1606.07792>`_ [1].

In general terms, ``pytorch-widedeep`` is a package to use deep learning with
tabular data. In particular, is intended to facilitate the combination of text
and images with corresponding tabular data using wide and deep models. With
that in mind there are two architectures that can be implemented with just a
few lines of code.


Architectures
-------------

**Architecture 1**

.. image:: figures/architecture_1.png
   :width: 600px
   :align: center

Architecture 1 combines the ``Wide``, one-hot encoded features with the
outputs from the ``DeepDense``, ``DeepText`` and ``DeepImage`` components
connected to a final output neuron or neurons, depending on whether we are
performing a binary classification or regression, or a multi-class
classification. The components within the faded-pink rectangles are
concatenated.

In math terms, and following the notation in the `paper
<https://arxiv.org/abs/1606.07792>`_, Architecture 1 can be formulated as:

.. image:: figures/architecture_1_math.png
   :width: 500px
   :align: center

Where *'W'* are the weight matrices applied to the wide model and to the final
activations of the deep models, *'a'* are these final activations, and
:math:`{\phi}` (x) are the cross product transformations of the original
features *'x'*. In case you are wondering what are *"cross product
transformations"*, here is a quote taken directly from the paper: *"For binary
features, a cross-product transformation (e.g., “AND(gender=female,
language=en)”) is 1 if and only if the constituent features (“gender=female”
and “language=en”) are all 1, and 0 otherwise".* Finally, :math:`{\sigma}` (.)
is the activation function.


**Architecture 2**

.. image:: figures/architecture_2.png
   :width: 600px
   :align: center

Architecture 2 combines the ``Wide`` one-hot encoded features with the Deep
components of the model connected to the output neuron(s), after the different
Deep components have been themselves combined through a FC-Head (referred as
as ``deephead``).

In math terms, and following the notation in the `paper
<https://arxiv.org/abs/1606.07792>`_, Architecture 2 can be formulated as:

.. image:: figures/architecture_2_math.png
   :width: 300px
   :align: center

When using ``pytorch-widedeep``, the assumption is that the so called ``Wide``
and ``DeepDense`` components in the figures are **always** present, while
``DeepText`` and ``DeepImage`` are optional. ``pytorch-widedeep`` includes
standard text (stack of LSTMs) and image (pre-trained ResNets or stack of
CNNs) models. However, the user can use any custom model as long as it has an
attribute called ``output_dim`` with the size of the last layer of
activations, so that ``WideDeep`` can be constructed. See the examples folder
for more information.

References
----------
[1] Heng-Tze Cheng, et al. 2016. Wide & Deep Learning for Recommender Systems.
`arXiv:1606.07792 <https://arxiv.org/abs/1606.07792>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
