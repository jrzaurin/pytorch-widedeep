def get_model_class(model_name: str):
    if "distilbert" in model_name:
        return "distilbert"
    elif "bert" in model_name:
        return "bert"
    elif "roberta" in model_name:
        return "roberta"
    elif "albert" in model_name:
        return "albert"
    elif "electra" in model_name:
        return "electra"
    else:
        raise ValueError(
            "model_name should belong to one of the following classes: "
            f"'distilbert', 'bert', 'roberta', 'albert', or 'electra'. Got {model_name}"
        )
