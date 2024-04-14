from typing import Tuple
from collections import namedtuple

from transformers import (
    BertModel,
    BertConfig,
    AlbertModel,
    AlbertConfig,
    ElectraModel,
    RobertaModel,
    BertTokenizer,
    ElectraConfig,
    RobertaConfig,
    AlbertTokenizer,
    DistilBertModel,
    PreTrainedModel,
    DistilBertConfig,
    ElectraTokenizer,
    PretrainedConfig,
    RobertaTokenizer,
    BertTokenizerFast,
    AlbertTokenizerFast,
    DistilBertTokenizer,
    PreTrainedTokenizer,
    ElectraTokenizerFast,
    RobertaTokenizerFast,
    DistilBertTokenizerFast,
)

ModelObjects = namedtuple(
    "ModelObjects", ["config", "tokenizer", "tokenizerfast", "model"]
)

MODEL_OBJECTS = {
    "distilbert": ModelObjects(
        DistilBertConfig,
        DistilBertTokenizer,
        DistilBertTokenizerFast,
        DistilBertModel,
    ),
    "bert": ModelObjects(BertConfig, BertTokenizer, BertTokenizerFast, BertModel),
    "roberta": ModelObjects(
        RobertaConfig, RobertaTokenizer, RobertaTokenizerFast, RobertaModel
    ),
    "albert": ModelObjects(
        AlbertConfig, AlbertTokenizer, AlbertTokenizerFast, AlbertModel
    ),
    "electra": ModelObjects(
        ElectraConfig, ElectraTokenizer, ElectraTokenizerFast, ElectraModel
    ),
}


def get_model_class(model_name: str):
    if "distilbert" in model_name:
        return "distilbert"
    elif "roberta" in model_name:
        return "roberta"
    elif "albert" in model_name:
        return "albert"
    elif "electra" in model_name:
        return "electra"
    elif "bert" in model_name:
        return "bert"
    else:
        raise ValueError(
            "model_name should belong to one of the following classes: "
            f"'distilbert', 'bert', 'roberta', 'albert', or 'electra'. Got {model_name}"
        )


def get_tokenizer(
    model_name: str, use_fast_tokenizer: bool = False, **kwargs
) -> PreTrainedTokenizer:
    model_class = get_model_class(model_name)
    if use_fast_tokenizer:
        return MODEL_OBJECTS[model_class].tokenizerfast.from_pretrained(
            model_name, **kwargs
        )
    else:
        return MODEL_OBJECTS[model_class].tokenizer.from_pretrained(
            model_name, **kwargs
        )


def get_config_and_model(
    model_name: str,
) -> Tuple[PretrainedConfig, PreTrainedModel]:
    model_class = get_model_class(model_name)
    config = MODEL_OBJECTS[model_class].config.from_pretrained(model_name)
    model = MODEL_OBJECTS[model_class].model(config)
    return config, model
