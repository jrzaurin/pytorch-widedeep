pytorch-widedeep
================

*A flexible package to combine tabular data with text and images using wide and deep models*

Below there is an introduction to the architectures one can build using
``pytorch-widedeep``. If you prefer to learn about the utilities and
components go straight to corresponding sections in the Documentation.

Documentation
-------------

.. toctree::
    :maxdepth: 1

    Installation <installation>
    Quick Start <quick_start>
    Utilities <utils/index>
    Preprocessing <preprocessing>
    Model Components <model_components>
    Metrics <metrics>
    Losses <losses>
    Dataloaders <dataloaders>
    Callbacks <callbacks>
    The Trainer <trainer>
    Tab2Vec <tab2vec>
    Examples <examples>


Introduction
------------
``pytorch-widedeep`` is based on Google's `Wide and Deep Algorithm
<https://arxiv.org/abs/1606.07792>`_.

In general terms, ``pytorch-widedeep`` is a package to use deep learning with
tabular and multimodal data. In particular, is intended to facilitate the
combination of text and images with corresponding tabular data using wide and
deep models. With that in mind there are a number of architectures that can
be implemented with just a few lines of code. The main components of those
architectures are shown in the Figure below:

.. image:: figures/widedeep_arch.png
   :width: 700px
   :align: center

The dashed boxes in the figure represent optional, overall components, and the
dashed lines indicate the corresponding connections, depending on whether or
not certain components are present. For example, the dashed, blue-arrows
indicate that the ``deeptabular``, ``deeptext`` and ``deepimage`` components
are connected directly to the output neuron or neurons (depending on whether
we are performing a binary classification or regression, or a multi-class
classification) if the optional ``deephead`` is not present. The components
within the faded-pink rectangle are concatenated.

Note that it is not possible to illustrate the number of possible
architectures and components available in ``pytorch-widedeep`` in one Figure.
Therefore, for more details on possible architectures (and more) please, read
this documentation, or see the `Examples
<https://github.com/jrzaurin/pytorch-widedeep/tree/master/examples>`_ folders
in the repo.

In math terms, and following the notation in the `paper
<https://arxiv.org/abs/1606.07792>`_, the expression for the architecture
without a ``deephead`` component can be formulated as:


.. image:: figures/architecture_1_math.png
   :width: 600px
   :align: center


Where *'W'* are the weight matrices applied to the wide model and to the final
activations of the deep models, :math:`a` are these final activations, and
:math:`{\phi}` (x) are the cross product transformations of the original
features *'x'*. In case you are wondering what are *"cross product
transformations"*, here is a quote taken directly from the paper: *"For binary
features, a cross-product transformation (e.g., “AND(gender=female,
language=en)”) is 1 if and only if the constituent features (“gender=female”
and “language=en”) are all 1, and 0 otherwise".* Finally, :math:`{\sigma}` (.)
is the activation function.

While if there is a ``deephead`` component, the previous expression turns
into:

.. image:: figures/architecture_2_math.png
   :width: 350px
   :align: center


It is important to emphasize that **each individual component, wide,
deeptabular, deeptext and deepimage, can be used independently and in
isolation**. For example, one could use only ``wide``, which is in simply a
linear model. In fact, one of the most interesting functionalities in
``pytorch-widedeep`` would be the use of the ``deeptabular`` component on its
own, i.e. what one might normally refer as Deep Learning for Tabular Data.
Currently, ``pytorch-widedeep`` offers the following different models for
that component:


1. **TabMlp**: a simple MLP that receives embeddings representing the
categorical features, concatenated with the continuous features.

2. **TabResnet**: similar to the previous model but the embeddings are
passed through a series of ResNet blocks built with dense layers.

3. **TabNet**: details on TabNet can be found in `TabNet: Attentive
Interpretable Tabular Learning <https://arxiv.org/abs/1908.07442>`_

And the ``Tabformer`` family, i.e. Transformers for Tabular data:

4. **TabTransformer**: details on the TabTransformer can be found in
`TabTransformer: Tabular Data Modeling Using Contextual Embeddings
<https://arxiv.org/pdf/2012.06678.pdf>`_.

5. **SAINT**: Details on SAINT can be found in `SAINT: Improved Neural
Networks for Tabular Data via Row Attention and Contrastive Pre-Training
<https://arxiv.org/abs/2106.01342>`_.

6. **FT-Transformer**: details on the FT-Transformer can be found in
`Revisiting Deep Learning Models for Tabular Data
<https://arxiv.org/abs/2106.11959>`_.

7. **TabFastFormer**: adaptation of the FastFormer for tabular data. Details
on the Fasformer can be found in `FastFormers: Highly Efficient Transformer
Models for Natural Language Understanding
<https://arxiv.org/abs/2010.13382>`_

8. **TabPerceiver**: adaptation of the Perceiver for tabular data. Details on
the Perceiver can be found in `Perceiver: General Perception with Iterative
Attention <https://arxiv.org/abs/2103.03206>`_

Note that while there are scientific publications for the TabTransformer,
SAINT and FT-Transformer, the TabFasfFormer and TabPerceiver are our own
adaptation of those algorithms for tabular data.

For details on these models and their options please see the examples in the
Examples folder and the documentation.

Finally, while I recommend using the ``wide`` and ``deeptabular`` models in
``pytorch-widedeep`` it is very likely that users will want to use their own
models for the ``deeptext`` and ``deepimage`` components. That is perfectly
possible as long as the the custom models have an attribute called
``output_dim`` with the size of the last layer of activations, so that
``WideDeep`` can be constructed. Again, examples on how to use custom
components can be found in the Examples folder. Just in case
``pytorch-widedeep`` includes standard text (stack of LSTMs or GRUs) and
image(pre-trained ResNets or stack of CNNs) models.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
