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

In recent years datasets have grown both in size and diversity, combining different data types or modes. In fact, it is not unusual these days to face machine learning projects that involve tabular data, images and/or text. Traditionally, one would address these projects by independently generating some features from every data mode, combine them at a later stage and then pass them to an algorithm such a GBM to perform classification or regression (if, for example, the nature of the problem is supervised).

However, with the advent of "_easy-to-use_" Deep Learning (DL) frames such as Tensorflow [@tensorflow2015-whitepaper] or Pytorch [@NEURIPS2019_9015], and the subsequent advances in the fields of Computer Vision, Natural Language Processing or Deep Learning for Tabular data, it is now possible to use SoTA DL models and combine all datasets earlier in the process. This has two main advantages: in the first place we can partially (or entirely) avoid the feature engineering step, which can be tedious in some cases. Secondly, the representations of each data mode, tabular, text and images, is learned jointly, harboring information that not only relates to the target, if the problem is supervised, but to how the different data modes relates to each other.

Furthermore, the flexibility inherent to using DL approaches allows the use of techniques that are principle designed for problems involving only text and/or images, to tabular data, such as for example transfer learning or self-supervised pre-training.

With that in mind we introduce `pytorch-widedeep`, a flexible package for multimodal-deep-learning, designed to facilitate the combination of tabular data with text and images.

# Statement of need

There is a small number of packages available to use DL for tabular data alone (e.g pytorch-tabular [@joseph2021pytorch], pytorch-tabnet or autogluon-tabular [@erickson2020autogluon]) or that focus mainly in combining text and images (e.g. MMF [@singh2020mmf]). With that in mind, our goal was to provide a modular, flexible, "_easy-to-use_" frame that allows the combination of a wide variety of models for all data modes, tabular, text and images. Furthermore, our library offers some in-house developed models for tabular data, such as adaptations of the Perceiver [@jaegle2021perceiver] and the FastFormer [@chuhan2021fastformer] that, to the best of our knowledge, are not available in any other of the packages mentioned before.

`pytorch-widedeep` is based on Google's Wide and Deep Algorithm [@cheng2016wide], and hence its name. However, the library has evolved enormously since its origin and, while we prefer to preserve the name for a variety of reasons (the explanation of which is beyond the scope of this paper), the original algorithm is heavily adjusted for multi-modal datasets and intended to facilitate the combination of text and images with corresponding tabular data. With that in mind there are a number of architectures that can be implemented with just a few lines of code. The main components of those architectures are shown in the Figure \autoref{fig:widedeep_arch}.

![\label{fig:widedeep_arch}](figures/widedeep_arch.png)

The blue and green boxes in the figure represent the main data modes and their corresponding model components, namely `wide`, `deeptabular`, `deeptext` and `deepimage`. The yellow boxes represent _so-called_ fully-connected (FC) heads, simply MLPs that one can optionally add _"on top"_ of the main components. These are referred in the figure as `TextHead` and `ImageHead`. The dashed-line rectangles indicate that the outputs from the components inside are concatenated if a final FC head (referred as `DeepHead` in the figure) is used. The faded-green `deeptabular` box aims to indicate that the output of the deeptabular component will be concatenated directly with the output of the `deeptext` or `deepimage` components or, alternatively, with the FC heads if these are used. Finally, the arrows indicate the connections, which of course depend on the final architecture that the user chooses to build. For example, if a model comprised by a `deeptabular` and a `deeptext` component with no FC heads is used, the outputs of those components will be directly "plugged" into the output neuron or neurons (depending on whether this is a regression, binary classification or multi-class classification problem).

In math terms, and following the notation in the original paper [@cheng2016wide], the expression for the architecture without a `deephead` component can be formulated as:


$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deeptabular}^{T}a_{deeptabular}^{l_f} + W_{deeptext}^{T}a_{deeptext}^{l_f} + W_{deepimage}^{l_f} + b)$


Where $W$ are the weight matrices applied to the wide model and to the final activations of the deep models, $a$ are these final activations, and $\phi(x)$ are the cross product transformations of the original features $x$. In case you are wondering what are _"cross product transformations"_, here is a quote taken directly from the paper: _"For binary features, a cross-product transformation (e.g., “AND(gender=female, language=en)”) is 1 if and only if the constituent features (“gender=female” and “language=en”) are all 1, and 0 otherwise"._


While if there is a `deephead` component, the previous expression turns into:


$pred = \sigma(W_{wide}^{T}[x,\phi(x)] + W_{deephead}^{T}a_{deephead}^{l_f} + b)$


At this stage it is worth mentioning that the library has been built with an special emphasis in flexibility. This is, we wanted users to easily run as many different models as possible and/or to use their custom components if they prefer. With that in mind, each and every data mode component in the figure above can be used independently and in isolation. For example, if the user wants to use a ResNet model to perform classification in an image-only dataset, that is perfectly possible using this library. In addition, following some minor adjustments described in the documentation, the user can use any custom model for each data mode -- mainly a custom model is a standard Pytorch model class that must have a property or attribute called `output_dim`. This way the `WideDeep` collector class knows the size of the incoming activations and is able to construct the multi-modal model --. Examples on how to use custom components can be found in the examples folder in the repo and in the examples section in the documentation of the package..


# The Model Hub

In this section we will describe the current model components available for each data mode in the library. Bear in mind that the library is constantly under development and models are constantly added to the "model-hub".

## The `wide` component

This is a linear model for tabular data where the non-linearities are captured via cross-product transformations (see the description in the previous section). This is the simplest of all components and we consider it a very useful as benchmark when used it on its own.


## The `deeptabular` component

Currently, `pytorch-widedeep` offers the following models for the so called `deeptabular` component:

1. **TabMlp**: a MLP that receives embeddings that are representations of the categorical features, concatenated with the continuous features, which can also be embedded.
2. **TabResnet**: similar to the `TabMlp` model, but the embeddings are passed through a series of ResNet blocks, that are built with dense layers.
3. **TabNet**: implementation of the TabNet [@arik2021tabnet] architecture. Our implementation is fully based on that at the [dreamquark repo](https://github.com/dreamquark-ai/tabnet). Therefore, all credit must go to their team and we strongly encourage the user to go and read their repo.

Two simple attention based models referred as:

4. **ContextAttentionMLP**: a MLP with at attention mechanism _"on top"_ that is based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical].
5. **SelfAttentionMLP**: a MLP with an attention mechanism that is a simplified version of a transformer block. We refer to this mechanism as "_query-key self-attention_". We implemented this model because we observed that the `TabTransformer`  [@huang2020tabtransformer] (also included in this library) has a notable tendency to over-fit.

The so-called `Tabformer` family, i.e. Transformers for Tabular data:

6. **TabTransformer**: our implementation of the TabTransformer [@huang2020tabtransformer]. Note that this is an enhanced implementation of the original model described in the paper. Please, see the documentation for details on the available parameters.
7. **SAINT**: implementation of the SAINT [@somepalli2021saint]. Similarly to the `TabTransformer`, this is an enhanced implementation of the original model described in the paper. Please, see the documentation for details on the available parameters.
8. **FT-Transformer**: implementation of the FT-Transformer [@gorishniy2021revisiting].
9. **TabFastFormer**: our adaptation of the FastFormer [@wu2021fastformer] for tabular data
10. **TabPerceiver**: our adaptation of the Perceiver [@jaegle2021perceiver] for tabular data

Probabilistic DL models for tabular data based on Weight Uncertainty in Neural Networks [@blundell2015weight]:

11. **BayesianWide**: a probabilistic adaptation of the `Wide` model described before.
12. **BayesianTabMlp**: a probabilistic adaptation of the `TabMlp` model described before.

Emphasize that while there are scientific publications for the `TabTransformer`, `SAINT` and `FT-Transformer`, the `TabFasfFormer` and `TabPerceiver` are our own adaptation of those algorithms for tabular data.

Overall, including the `wide` component, there are 13 DL-based models for tabular data available in the library. For details on these models, their corresponding options and examples of third party integrations please see the Examples [@pytorch_widedeep_examples].

## The `deepimage` component

The image related component is fully integrated with the newest version torchvision [@torchvision_models] (0.13 at the time of writing). This version has Multi-Weight Support [@torchvision_weight]. Therefore, a variety of model variants are available to use with pre-trained weights obtained with different datasets. Currently, the model variants supported by `pytorch-widedeep` are:

1. Resnet [@he2016deep]
2. Shufflenet [@zhang2018shufflenet]
3. Resnext [@xie2017aggregated]
4. Wide Resnet [@zagoruyko2016wide]
5. Regnet [@xu2022regnet]
6. Densenet [@huang2017densely]
7. Mobilenet [@howard2017mobilenets]
8. MNasnet [@tan2019mnasnet]
9. Efficientnet [tan2019efficientnet]
10. Squeezenet [@iandola2016squeezenet]

For details on this models please see the corresponding publications. For details on the pre-trained weights available please the torchvision  [@torchvision_models] docs.

## The `deeptext` component

Currently, `pytorch-widedeep` offers the following models for the `deeptext` component:

1. BasicRNN: a basic RNN that can be an LSTM [@hochreiter1997lstm] or a GRU [@cho2014properties] with their usual, corresponding parameters
2. AttentiveRNN: a basic RNN with an attention mechanism that is based on Hierarchical Attention Networks for Document Classification [@yang2016hierarchical].
3. StackedAttentiveRNN: a stack of `AttentiveRNN`s


At this stage it is clear that the model component for the text data mode is perhaps the weakest of all three. This is because even though currently we allow to use pre-trained word vectors, `pytorch-widedeep` does not support fully pre-trained models, such as BERT [@devlin2018bert] and its "family" (e.g. Roberta [@liu2019roberta]) "_out of the box_". The reason behind this design decision is that, initially, when we started building the library we thought that combining tabular data and text and image datasets using complex components for all different data modes might lead to intractable models.

However, a lot has changed in the field since then and it is now our priority to integrate `pytorch-widedeep` with the fantastic Hugginface library [@wolf2019huggingface]. At that point fully pre-trained models will be available for all data modes within the library.

Nonetheless, it is worth emphasizing that custom components can be easily used with the library, i.e. the user can externally define a NLP model with pre-trained weights, and combine it with any of the other model components using the `WideDeep` class (see the Examples section in the docs for details).

# Forms of model training:

Training single- or multi-mode models in `pytorch-widedeep` is handled by the different training classes. Currently, `pytorch-widedeep` offers the following training options:

1. "_Standard_" Supervised training
2. Supervised Bayesian or probabilistic training
3. Self-supervised pre-training

## Standard Supervised training

Standard supervised training for multi-modal models is handled in `pytorch-widedeep` by the `Trainer` class. Following the library's philosophy (simplicity and flexibility), the `Trainer` is designed to offer a large number of options to train models. For example, the library comes with a number of available loss functions, however, these might not fit the purposes of the user. With that in mind, a custom loss function can be easily passed to the `Trainer` via the `custom_loss_function` parameter.

In addition, it is possible to use any optimizer or learning rate scheduler available in `Pytorch`, warm up the different model components or fine-tune existing models via the `warmup` or `finetune` parameter, etc. Please, see the documentation for these and all the other options available for supervised training.

## Supervised Bayesian or probabilistic training

Supervised, probabilistic training is handled in `pytorch-widedeep` by the `BayesianTrainer` class. Note that probabilistic training in the library is only supported for tabular data. Also both the Bayesian models mentioned before and the `BayesianTrainer` class are inspired by the publication Weight Uncertainty in Neural Networks[@blundell2015weight], by Blundell et al.

As the `Trainer` class, the `BayesianTrainer` is designed to be flexible. However, given the probabilistic nature of the process, the amount of parameters and training options is lower than that of the `Trainer`.

On the other hand, it is worth mentioning that the documentation of the class is aimed to be self-contained, so that the user can utilize the `BayesianTrainer` using only the docs in the library. However, we strongly recommend reading the paper to fully understand the meaning of the training parameters and adequately setting their values.

## Self-Supervised pre-training

As in the case of Bayesian training, Self-Supervised pre-training is only supported for tabular data.

We have implemented two methods or routines that allow the user to self-supervised pre-training for all tabular models in the library with the exception of the `TabPerceiver`. This is because the `TabPerceiver` is a particular model it requires some adjustments when self-supervised pre-training. Such adjustments will be implemented in future versions of the library.

The two training routines available in the library are exposed to the user via the `EncoderDecoderTrainer` class and the `ContrastiveDenoisingTrainer` class. The former is based on TabNet: Attentive Interpretable Tabular Learning [@arik2021tabnet] and is illustrated in their Figure 2. The routine executed by the `EncoderDecoderTrainer` class can be described as a _standard_  encoder-decoder architecture, and is designed for models that are not "transformer-based", or models for which the embeddings can all have different dimensions.

The `ContrastiveDenoisingTrainer` is based on SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training [@somepalli2021saint] and is illustrated in the Figure 1 in their publication. The routine executed by the `ContrastiveDenoisingTrainer` is based on contrastive and denoising learning (see the original publication and references therein for details on these techniques), and is designed for models that are "transformer-based", or models for which the embeddings must all have the same dimensions. Describing this routine in detail is beyond the scope of this publication and we highly encourage the user of the library to read the original publication.

Please see the Examples [@pytorch_widedeep_examples] folder in the repo or the examples section in the documentation for details on how to use self-supervised pre-training within the library. Finally, it is worth mentioning that to fully utilize the self-supervised trainers mentioned before (the `EncoderDecoderTrainer` and the `ContrastiveDenoisingTrainer`), a minimum understanding of the processes as described in the papers is required. Therefore, we strongly encourage the users and reader to follow the related papers explanations.
  
# Contribution

`Pytorch-widedeep` is being developed and used by many active community members. Anyone can join the discussion on slack [@pytorch_widedeep_slack]

# Acknowledgements

We acknowledge work of other researchers, engineers and programmers from following projects and libraries:

* the `Callbacks` and `Initializers` structure and code is inspired by the torchsample library [@torch_sample], which in itself partially inspired by Keras [@keras]
* the `TextProcessor` class in this library uses the fastai [@fastai_tokenizer] `Tokenizer` and `Vocab`; the code at `utils.fastai_transforms` is a minor adaptation of their code so it functions within this library; to our experience their `Tokenizer` is the best in class
* the `ImageProcessor` class in this library uses code from the fantastic Deep Learning for Computer Vision (DL4CV) [@dl4cv] book by Adrian Rosebrock
* we adjusted and integrated ideas of Label and Feature Distribution Smoothing [@yang2021delving]
* we adjusted and integrated ZILNloss code written in Tensorflow/Keras [@wang2019deep]

# References
