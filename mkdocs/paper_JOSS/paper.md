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
 - name: Centre Tecnologic de Telecomunicacions de Catalunya (CTTC/CERCA), Catalunya, Spain
   index: 2
date: 30 July 2022
bibliography: paper.bib
---

# Summary

We introduce a flexible package for multimodal-deep-learning to combine tabular data with text and images using Wide and Deep models in Pytorch.

# Statement of need

``pytorch-widedeep`` is based on Google's Wide and Deep Algorithm [@cheng2016wide], adjusted for multi-modal datasets.

In general terms, `pytorch-widedeep` is a package to use deep learning with tabular data. In particular, is intended to facilitate the combination of text and images with corresponding tabular data using wide and deep models. With that in mind there are a number of architectures that can be implemented with just a few lines of code. The main components of those architectures are shown in the Figure below:

<p align="center">
  <img width="750" src="docs/figures/widedeep_arch.png">
</p>

The dashed boxes in the figure represent optional, overall components, and the dashed lines/arrows indicate the corresponding connections, depending on whether or not certain components are present. For example, the dashed, blue-lines indicate that the ``deeptabular``, ``deeptext`` and ``deepimage`` components are connected directly to the output neuron or neurons (depending on whether we are performing a binary classification or regression, or a multi-class classification) if the optional ``deephead`` is not present. Finally, the components within the faded-pink rectangle are concatenated.

Note that it is not possible to illustrate the number of possible architectures and components available in ``pytorch-widedeep`` in one Figure. Therefore, for more details on possible architectures (and more) please, read this documentation, or see the Examples [@pytorch_widedeep_examples].

In math terms, and following the notation in the paper [@cheng2016wide], the expression for the architecture without a ``deephead`` component can be formulated as:

$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deeptabular}^{T}a_{deeptabular}^{l_f} + W_{deeptext}^{T}a_{deeptext}^{l_f} + W_{deepimage}^{l_f} + b)$

Where *'W'* are the weight matrices applied to the wide model and to the final activations of the deep models, *'a'* are these final activations, and &phi;(x) are the cross product transformations of the original features *'x'*. In case you are wondering what are *"cross product transformations"*, here is a quote taken directly from the paper: *"For binary features, a cross-product transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if the constituent features (“gender=female” and “language=en”) are all 1, and 0 otherwise".*


While if there is a ``deephead`` component, the previous expression turns into:

$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deephead}^{T}a_{deephead}^{l_f} + b)$

It is perfectly possible to use custom models (and not necessarily those in the library) as long as the the custom models have an attribute called ``output_dim`` with the size of the last layer of activations, so that ``WideDeep`` can be constructed. Examples on how to use custom components can be found in the Examples [@pytorch_widedeep_examples].


# Components

It is important to emphasize that **each individual component, `wide`, `deeptabular`, `deeptext` and `deepimage`, can be used independently** and in isolation. For example, one could use only `wide`, which is in simply a linear model. In fact, one of the most interesting functionalities in``pytorch-widedeep`` would be the use of the ``deeptabular`` component on its own, i.e. what one might normally refer as Deep Learning for Tabular Data.

## The ``deeptabular`` component

Currently, ``pytorch-widedeep`` offers the following different models for the ``deeptabular`` component:

0. **Wide**: a simple linear model where the nonlinearities are captured via cross-product transformations, as explained before.
1. **TabMlp**: a simple MLP that receives embeddings representing the categorical features, concatenated with the continuous features, which can also be embedded.
2. **TabResnet**: similar to the previous model but the embeddings are passed through a series of ResNet blocks built with dense layers.
3. **TabNet**: implementation of the TabNet [@arik2021tabnet]

Two simpler attention based models that we call:

4. **ContextAttentionMLP**: MLP with at attention mechanism "on top" that is based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical]
5. **SelfAttentionMLP**: MLP with an attention mechanism that is a simplified version of a transformer block that we refer as "query-key self-attention".

The ``Tabformer`` family, i.e. Transformers for Tabular data:

1. **TabTransformer**: implementation of the TabTransformer [@huang2020tabtransformer]
2. **SAINT**: implementation of the SAINT [@somepalli2021saint]
3. **FT-Transformer**: implementation of the FT-Transformer [@gorishniy2021revisiting].
4. **TabFastFormer**: adaptation of the FastFormer [@kim2020fastformers]
5. **TabPerceiver**: adaptation of the Perceiver [@jaegle2021perceiver] for tabular data

And probabilistic DL models for tabular data based on Weight Uncertainty in Neural Networks [@blundell2015weight]:

11. **BayesianWide**: probabilistic adaptation of the `Wide` model.
12. **BayesianTabMlp**: probabilistic adaptation of the `TabMlp` model

Note that while there are scientific publications for the TabTransformer, SAINT and FT-Transformer, the TabFasfFormer and TabPerceiver are our own adaptation of those algorithms for tabular data.

For details on these models (and all the other models in the library for the different data modes), their corresponding options and examples of third party integrations please see the Examples [@pytorch_widedeep_examples].

## The ``deeptext`` component

Currently, ``pytorch-widedeep`` offers the following models for the ``deeptext`` component:
* BasicRNN
* AttentiveRNN
* StackedAttentiveRNN
    
The last two are based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical].


## The ``deepimage`` component

The image related component is fully integrated with the torchvision [@torchvision_models], with a Multi-Weight Support API [@torchvision_weight]. Currently, the model variants supported by ``pytorch-widedeep`` are: 
* resnet [@resnet]
* shufflenet [@shufflenet]
* resnext [@resnext]
* wide_resnet [@wide_resnet]
* regnet [@regnet]
* densenet [@densenet]
* mobilenet [@mobilenetv2] [@mobilenetv3]
* mnasnet [@mnasnet]
* efficientnet [@efficientnet] [@efficientnetv2]
* squeezenet [@squeezenet]

# Forms of model training:

Currently, ``pytorch-widedeep`` offers the following methods of model training:
* supervised training
* bayesian or probabilistic training, inspired by the paper Weight Uncertainty in Neural Networks[@blundell2015weight]
* self-supervised pre-training

We believe supervised and bayesian training do not need additional explanation and in the following we focus on self-supervised pre-training.

## Self Supervised Pre-training for tabular data

We have implemented two methods or routines that allow the user to self-suerpvised pre-training for all tabular models in the library with the exception of the `TabPerceiver` (this is a particular model and self-supervised pre-training requires some adjustments that will be implemented in future versions). Please see the Examples [@pytorch_widedeep_examples] or the examples section in the documentation for details on how to use self-supervised pre-training with this library.

The two routines implemented are illustrated in the figures below. The first is from TabNet [@arik2021tabnet]. It is a *'standard'* encoder-decoder architecture and and is designed here for models that do not use transformer-based architectures (or when the embeddings can all have different dimensions). The second is from SAINT [@somepalli2021saint], it is based on Contrastive and Denoising learning and is designed for models that use transformer-based architectures (or when the embeddings all need to have the same dimension):

<p align="center">
  <img width="750" src="../docs/figures/self_supervised_tabnet.png">
</p>

Figure 1. Figure 2 in their paper [@arik2021tabnet]. The caption of the original paper is included in case it is useful.

<p align="center">
  <img width="700" src="../docs/figures/self_supervised_saint.png">
</p>

Figure 2. Figure 1 in their paper [@gorishniy2022embeddings]. The caption of the original paper is included in case it is useful.

To fully utilise the self-supervised trainers implemented in this library a minimum understanding of the processes as described in the papers is required. Therefore, we strongly encourage the users and reader to follow the related papers explanations.
  
# Contribution

Pytorch-widedeep is being developed and used by many active community members. Anyone can join the dicussion on slack [@pytorch_widedeep_slack]

# Acknowledgements

We acknowledge work of other researchers, engineers and programmers from following projects and libraries:

* the `Callbacks` and `Initializers` structure and code is inspired by the torchsample library [@torchsample], which in itself partially inspired by Keras [@keras]
* the `TextProcessor` class in this library uses the fastai [@fastai_tokenizer] `Tokenizer` and `Vocab`; the code at `utils.fastai_transforms` is a minor adaptation of their code so it functions within this library; to our experience their `Tokenizer` is the best in class
* the `ImageProcessor` class in this library uses code from the fantastic Deep Learning for Computer Vision (DL4CV) [@dl4cv] book by Adrian Rosebrock
* we adjusted and integrated ideas of Label and Feature Distribution Smoothing [@yang2021delving]
* we adjusted and integrated ZILNloss code written in Tensorflow/Keras [@wang2019deep]

# References
