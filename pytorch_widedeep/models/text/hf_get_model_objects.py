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
    DistilBertModel,
    DistilBertConfig,
    AlbertTokenizerFast,
    ElectraTokenizerFast,
    RobertaTokenizerFast,
    DistilBertTokenizerFast,
)


def get_model_objects(model_name: str, model_type: str):
    if model_type == "distilbert":
        config = DistilBertConfig.from_pretrained(model_name)
        tokenizer = DistilBertTokenizerFast(config)
        model = DistilBertModel(config)
    elif model_type == "bert":
        config = BertConfig.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel(config)
    elif model_type == "roberta":
        config = RobertaConfig.from_pretrained(model_name)
        tokenizer = RobertaTokenizerFast(config)
        model = RobertaModel(config)
    elif model_type == "albert":
        config = AlbertConfig.from_pretrained(model_name)
        tokenizer = AlbertTokenizerFast(config)
        model = AlbertModel(config)
    elif model_type == "electra":
        config = ElectraConfig.from_pretrained(model_name)
        tokenizer = ElectraTokenizerFast(config)
        model = ElectraModel(config)
    else:
        raise ValueError(
            f"model_type should be one of 'distilbert', 'bert', 'roberta', 'albert', or 'electra'. Got {model_type}"
        )
    return config, tokenizer, model
