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
date: 03 September 2022
bibliography: paper.bib
---

# Summary

In recent years datasets have grown both in size and also diversity, combining different data types (or modes). In fact, it is not unusual these days to face machine learning projects that involve tabular data, images and/or text. Traditionally, one would address these projects by independently generating some features from every data mode, combine them at a later stage and then pass them to an algorithm such a GBMs to perform classification or regression if, for example, the nature of the problem is supervised.

However, with the advent of "_easy-to-use_" Deep Learning (DL) frames such as Tensorflow or Pytorch, and the subsequent advances in the fields of Computer Vision, Natural Language Processing and Deep Learning for Tabular data, it is possible to use SoTA DL models and combine all datasets earlier in the process. This has two main advantages: in the first place we can partially (or entirely) avoid the feature engineering step, which can be tedious in some cases. Secondly, the representations of each data mode, tabular, text and images, is learned jointly, harboring information that not only relates to the target, if the problem is supervised, but to how the different data modes dataset relates to each other.

Furthermore, the flexibility inherent to using DL approaches allows the use of techniques that are principle designed for problems involving only text and/or images, to tabular data, such as for example transfer learning or self-supervised pre-training.

With that in mind we introduce pytorch-widedeep, a flexible package for multimodal-deep-learning, designed to facilitate the combination of tabular data with text and images.

# Statement of need

There is a small number of packages available to use DL for tabular data alone ([insert ref for pytorch-tabular, tabnet and autogluon tabular]) or that focus mainly in combining text and images ([insert ref to torch multimodal]). With that in mind, our goal was to provide a modular, flexible, "_easy-to-use_" frame that allows the combination of a wide variety of models for all data modes, tabular, text and images. Furthermore, our library offers some in-house developed models for tabular data (such as adaptations of the Perceiver and the FastFormer. [insert ref here]) that, to the best of our knowledge, are not available in any other of the packages mentioned before.

`pytorch-widedeep` is based on Google's Wide and Deep Algorithm [@cheng2016wide], and hence its name. However, the library has evolved enormously since its origin and, while we prefer to preserve the name for a variety of reasons (the explanation of which is beyond the scope of this paper), the original algorithm is heavily adjusted for multi-modal datasets and intended to facilitate the combination of text and images with corresponding tabular data. With that in mind there are a number of architectures that can be implemented with just a few lines of code. The main components of those architectures are shown in the Figure \autoref{fig:widedeep_arch} (NEED TO UPDATE FIGURE).

![\label{fig:widedeep_arch}](figures/widedeep_arch.png)


The dashed boxes in the figure represent optional model components for a given data mode, and the dashed lines/arrows indicate the corresponding connections, depending on whether or not those optional components are present. For example, the dashed, blue-lines indicate that the `deeptabular`, `deeptext` and `deepimage` components are connected directly to the output neuron or neurons (depending on whether we are performing a binary classification or regression, or a multi-class classification) if an optional `deephead` is not present. The components within the faded-pink rectangle are concatenated. Of course, it is not possible to illustrate the number of possible architectures and components available in `pytorch-widedeep` in one Figure. Therefore, for more details on possible architectures (and more) please see the Examples [@pytorch_widedeep_examples] folder in the repo or the Examples section in the documentation of the package.

In math terms, and following the notation in the original paper [@cheng2016wide], the expression for the architecture without a `deephead` component can be formulated as:


$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deeptabular}^{T}a_{deeptabular}^{l_f} + W_{deeptext}^{T}a_{deeptext}^{l_f} + W_{deepimage}^{l_f} + b)$


Where $W$ are the weight matrices applied to the wide model and to the final activations of the deep models, $a$ are these final activations, and $\phi(x)$ are the cross product transformations of the original features $x$. In case you are wondering what are _"cross product transformations"_, here is a quote taken directly from the paper: _"For binary features, a cross-product transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if the constituent features (“gender=female” and “language=en”) are all 1, and 0 otherwise"._


While if there is a `deephead` component, the previous expression turns into:


$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deephead}^{T}a_{deephead}^{l_f} + b)$

At this stage it is worth mentioning that the library has been built with an special emphasis in flexibility. This is, we wanted users to easily run as many different models as possible and/or to use their custom components if they prefer. With that in mind, each and every data mode component in the figure above can be used independently and in isolation. For example, if the user wants to use a ResNet model to perform classification in an image-only dataset, that is perfectly possible using this library. In addition, following some minor adjustments described in the documentation, the user can use any custom model for each data mode -- mainly a custom model is a standard pytorch model class that must have a property or attribute called `output_dim`. This way the `WideDeep` collector class knows the incoming activations and is able to construct the multimodal model --. Examples on how to use custom components can be found in the Examples [@pytorch_widedeep_examples] folder in the repo and in the Examples section in the documentation of the package..


# The Model Hub

In this section we will describe the current model components available for each data mode in the library. Bear in mind that the library is constantly under development and models are constantly added to the "model-hub"

## The `deeptabular` component

Currently, `pytorch-widedeep` offers the following models for the so called `deeptabular` component:

1. **Wide**: a simple linear model where the non-linearities are captured via cross-product transformations. This is the simplest of all components and we find it very useful as a benchmark model when used it on its own.
2. **TabMlp**: a MLP that receives embeddings that are representations of the categorical features, concatenated with the continuous features, which can also be embedded.
3. **TabResnet**: similar to the `TabMlp` model, but the embeddings are passed through a series of ResNet blocks, that are built with dense layers.
4. **TabNet**: implementation of the TabNet [@arik2021tabnet]. Our implementation is fully based on that at the dreamquark repo. Therefore, all credit must go to their team and we strongly encourage the user to go and read their repo.

Two simpler attention based models referred as:

5. **ContextAttentionMLP**: a MLP with at attention mechanism "on top" that is based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical].
6. **SelfAttentionMLP**: a MLP with an attention mechanism that is a simplified version of a transformer block. We refer to this mechanism as "_query-key self-attention_". We implemented this model because we observed that the `TabTransformer`  [@huang2020tabtransformer] (also included in this library) has a notable tendency to over-fit.

The so-alled `Tabformer` family, i.e. Transformers for Tabular data:

7. **TabTransformer**: our implementation of the TabTransformer [@huang2020tabtransformer]. Note that this is an enhanced implementation of the original model described in the paper. Please, see the documentation for details on the available parameters.
8. **SAINT**: implementation of the SAINT [@somepalli2021saint]. Similarly to the `TabTransformer`, this is an enhanced implementation of the original model described in the paper. Please, see the documentation for details on the available parameters.
9. **FT-Transformer**: implementation of the FT-Transformer [@gorishniy2021revisiting].
10. **TabFastFormer**: our adaptation of the FastFormer [@kim2020fastformers] for tabular data
11. **TabPerceiver**: our adaptation of the Perceiver [@jaegle2021perceiver] for tabular data

Probabilistic DL models for tabular data based on Weight Uncertainty in Neural Networks [@blundell2015weight]:

12. **BayesianWide**: a probabilistic adaptation of the `Wide` model described before.
13. **BayesianTabMlp**: a probabilistic adaptation of the `TabMlp` model described before.

In addition, Self-Supervised pre-training can be used for all `deeptabular` models, with the exception of the probabilistic models and the `TabPerceiver`, which we will describe in detail in Section X.

Emphasize that while there are scientific publications for the TabTransformer, SAINT and FT-Transformer, the TabFasfFormer and TabPerceiver are our own adaptation of those algorithms for tabular data.

FOverall, the user has 13 DL-based models for tabular data. For details on these models, their corresponding options and examples of third party integrations please see the Examples [@pytorch_widedeep_examples].

## The `deepimage` component

The image related component is fully integrated with the newest version torchvision [@torchvision_models] (0.13 at the time of writing). This version has Multi-Weight Support [@torchvision_weight]. Therefore, a variety of model variants are available to use with pre-trained weights obtained with different datasets. Currently, the model variants supported by `pytorch-widedeep` are:

1. Resnet [@resnet]
2. Shufflenet [@shufflenet]
3. Resnext [@resnext]
4. Wide Resnet [@wide_resnet]
5. Regnet [@regnet]
6. Densenet [@densenet]
7. Mobilenet [@mobilenetv2] [@mobilenetv3]
8. MNasnet [@mnasnet]
9. Efficientnet [@efficientnet] [@efficientnetv2]
10. Squeezenet [@squeezenet]

For details on this models please see the corresponding publications. For details on the pre-trained weights available please the torchvision  [@torchvision_models] docs.

## The `deeptext` component

Currently, `pytorch-widedeep` offers the following models for the `deeptext` component:

1. BasicRNN: a basic RNN that can be an LSTM or a GRU with their usual, corresponding parameters
2. AttentiveRNN: a basic RNN with an attention mechanism that is based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical].
3. StackedAttentiveRNN: a stack of `AttentiveRNN`s


At this stage it is clear that the model component for the text data mode is perhaps the weakest of all three. This is because even though currently we allow to use pre-trained word vectors, `pytorch-widedeep` does not support fully pre-trained models (e.g. the BERT family [insert ref]) "_out of the box_". The reason behind this design decision is that, initially, when we started building the library we thought that combining tabular, data and image datasets using complex components for all different data modes might lead to intractable models.

However, a lot has changed in the field since then and it is now our priority to integrate `pytorch-widedeep` with the fantastic Hugginface library ([insert ref]). At that point fully pre-trained models will be availble for all data modes within the library.

Nonetheless, it is worth emphasizing that custom components can be easily used with the library, i.e. the user can externally define a NLP model with pre-trained weights, and combine it with any of the other model components using the `WideDeep` class (see the Examples section in the docs for details).

# Forms of model training:

Currently, `pytorch-widedeep` offers the following methods of model training:

1. supervised training
2. bayesian or probabilistic training, inspired by the paper Weight Uncertainty in Neural Networks[@blundell2015weight]
3. self-supervised pre-training

We believe supervised and bayesian training do not need additional explanation and in the following we focus on self-supervised pre-training.

## Self Supervised Pre-training for tabular data

We have implemented two methods or routines that allow the user to self-suerpvised pre-training for all tabular models in the library with the exception of the `TabPerceiver` (this is a particular model and self-supervised pre-training requires some adjustments that will be implemented in future versions). Please see the Examples [@pytorch_widedeep_examples] or the examples section in the documentation for details on how to use self-supervised pre-training with this library.

The two routines implemented are illustrated in the Figures \autoref{fig:self_supervised_tabnet} \autoref{fig:self_supervised_saint}. The first is from TabNet [@arik2021tabnet]. It is a *'standard'* encoder-decoder architecture and and is designed here for models that do not use transformer-based architectures (or when the embeddings can all have different dimensions). The second is from SAINT [@somepalli2021saint], it is based on Contrastive and Denoising learning and is designed for models that use transformer-based architectures (or when the embeddings all need to have the same dimension):

![The caption of the original paper[@arik2021tabnet] is included in case it is useful. \label{fig:self_supervised_tabnet}](figures/self_supervised_tabnet.png)

![The caption of the original paper[@somepalli2021saint] is included in case it is useful. \label{fig:self_supervised_saint}](figures/self_supervised_saint.png)


To fully utilise the self-supervised trainers implemented in this library a minimum understanding of the processes as described in the papers is required. Therefore, we strongly encourage the users and reader to follow the related papers explanations.
  
# Contribution

Pytorch-widedeep is being developed and used by many active community members. Anyone can join the dicussion on slack [@pytorch_widedeep_slack]

# Acknowledgements

We acknowledge work of other researchers, engineers and programmers from following projects and libraries:

* the `Callbacks` and `Initializers` structure and code is inspired by the torchsample library [@torch_sample], which in itself partially inspired by Keras [@keras]
* the `TextProcessor` class in this library uses the fastai [@fastai_tokenizer] `Tokenizer` and `Vocab`; the code at `utils.fastai_transforms` is a minor adaptation of their code so it functions within this library; to our experience their `Tokenizer` is the best in class
* the `ImageProcessor` class in this library uses code from the fantastic Deep Learning for Computer Vision (DL4CV) [@dl4cv] book by Adrian Rosebrock
* we adjusted and integrated ideas of Label and Feature Distribution Smoothing [@yang2021delving]
* we adjusted and integrated ZILNloss code written in Tensorflow/Keras [@wang2019deep]

# References
