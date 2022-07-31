---
title: 'pytorch-widedeep: A flexible package for multimodal-deep-learning'
tags:
  - Python
  - Pytorch
  - Deep learning
authors:
  - name: Javier Rodriguez Zaurin
    orcid: 0000-0000-0000-0000
    # equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Pavol Mulinka
    orcid: 0000-0002-9394-8794
    # equal-contrib: true
    affiliation: 2
affiliations:
 - name: Independent Researcher, Spain
   index: 1
 - name: Centre Tecn\`ologic de Telecomunicacions de Catalunya (CTTC/CERCA), Catalunya, Spain
   index: 2
date: 30 July 2022
bibliography: paper.bib
---

# Summary

A flexible package for multimodal-deep-learning to combine tabular data with
text and images using Wide and Deep models in Pytorch.

# Statement of need

``pytorch-widedeep`` is based on Google's Wide and Deep Algorithm [@cheng2016wide],
adjusted for multi-modal datasets

In general terms, `pytorch-widedeep` is a package to use deep learning with
tabular data. In particular, is intended to facilitate the combination of text
and images with corresponding tabular data using wide and deep models. With
that in mind there are a number of architectures that can be implemented with
just a few lines of code. The main components of those architectures are shown
in the Figure below:


<p align="center">
  <img width="750" src="/docs/figures/widedeep_arch.png">
</p>

The dashed boxes in the figure represent optional, overall components, and the
dashed lines/arrows indicate the corresponding connections, depending on
whether or not certain components are present. For example, the dashed,
blue-lines indicate that the ``deeptabular``, ``deeptext`` and ``deepimage``
components are connected directly to the output neuron or neurons (depending
on whether we are performing a binary classification or regression, or a
multi-class classification) if the optional ``deephead`` is not present.
Finally, the components within the faded-pink rectangle are concatenated.

Note that it is not possible to illustrate the number of possible
architectures and components available in ``pytorch-widedeep`` in one Figure.
Therefore, for more details on possible architectures (and more) please, see
the
[documentation]((https://pytorch-widedeep.readthedocs.io/en/latest/index.html)),
or the Examples folders and the notebooks there.

In math terms, and following the notation in the
paper [@cheng2016wide], the expression for the architecture
without a ``deephead`` component can be formulated as:

$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deeptabular}^{T}a_{deeptabular}^{l_f} + W_{deeptext}^{T}a_{deeptext}^{l_f} + W_{deepimage}^{l_f} + b)$

Where *'W'* are the weight matrices applied to the wide model and to the final
activations of the deep models, *'a'* are these final activations, and
&phi;(x) are the cross product transformations of the original features *'x'*.
In case you are wondering what are *"cross product transformations"*, here is
a quote taken directly from the paper: *"For binary features, a cross-product
transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if
the constituent features (“gender=female” and “language=en”) are all 1, and 0
otherwise".*


While if there is a ``deephead`` component, the previous expression turns
into:

$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deephead}^{T}a_{deephead}^{l_f} + b)$

It is perfectly possible to use custom models (and not necessarily those in
the library) as long as the the custom models have an attribute called
``output_dim`` with the size of the last layer of activations, so that
``WideDeep`` can be constructed. Examples on how to use custom components can
be found in the Examples folder.

### The ``deeptabular`` component

It is important to emphasize that **each individual component, `wide`,
`deeptabular`, `deeptext` and `deepimage`, can be used independently** and in
isolation. For example, one could use only `wide`, which is in simply a
linear model. In fact, one of the most interesting functionalities
in``pytorch-widedeep`` would be the use of the ``deeptabular`` component on
its own, i.e. what one might normally refer as Deep Learning for Tabular
Data. Currently, ``pytorch-widedeep`` offers the following different models
for that component:

0. **Wide**: a simple linear model where the nonlinearities are captured via
cross-product transformations, as explained before.
1. **TabMlp**: a simple MLP that receives embeddings representing the
categorical features, concatenated w the continuous features, which can
also be embedded.
2. **TabResnet**: similar to the previous model but the embeddings are
passed through a series of ResNet blocks built with dense layers.
3. **TabNet**: details on TabNet can be found in
TabNet: Attentive Interpretable Tabular Learning [@arik2021tabnet]

The ``Tabformer`` family, i.e. Transformers for Tabular data:

4. **TabTransformer**: details on the TabTransformer can be found in
TabTransformer: Tabular Data Modeling Using Contextual Embeddings [@huang2020tabtransformer].
5. **SAINT**: Details on SAINT can be found in
SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training [@somepalli2021saint].
6. **FT-Transformer**: details on the FT-Transformer can be found in
Revisiting Deep Learning Models for Tabular Data [@gorishniy2021revisiting].
7. **TabFastFormer**: adaptation of the FastFormer for tabular data. Details
on the Fasformer can be found in
FastFormers: Highly Efficient Transformer Models for Natural Language Understanding [@kim2020fastformers]
8. **TabPerceiver**: adaptation of the Perceiver for tabular data. Details on
the Perceiver can be found in
Perceiver: General Perception with Iterative Attention [@jaegle2021perceiver]

And probabilistic DL models for tabular data based on
Weight Uncertainty in Neural Networks [@blundell2015weight]:

1. **BayesianWide**: Probabilistic adaptation of the `Wide` model.
2.  **BayesianTabMlp**: Probabilistic adaptation of the `TabMlp` model

Note that while there are scientific publications for the TabTransformer,
SAINT and FT-Transformer, the TabFasfFormer and TabPerceiver are our own
adaptation of those algorithms for tabular data.


# Acknowledgements

We acknowledge work of other researchers, engineers and programmers from following
projects and libraries:

* The `Callbacks` and `Initializers` structure and code is inspired by the
[`torchsample`](https://github.com/ncullen93/torchsample) library, which in
itself partially inspired by [`Keras`](https://keras.io/).
* The `TextProcessor` class in this library uses the
[`fastai`](https://docs.fast.ai/text.transform.html#BaseTokenizer.tokenizer)'s
`Tokenizer` and `Vocab`. The code at `utils.fastai_transforms` is a minor
adaptation of their code so it functions within this library. To my experience
their `Tokenizer` is the best in class.
* The `ImageProcessor` class in this library uses code from the fantastic [Deep
Learning for Computer
Vision](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)
(DL4CV) book by Adrian Rosebrock.
* We adjusted and integrated ideas of Label and Feature Distribution Smoothing [@yang2021delving]

# References