import os
from typing import List, Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
from transformers import (
    BertTokenizer,
    AlbertTokenizer,
    ElectraTokenizer,
    RobertaTokenizer,
    BertTokenizerFast,
    AlbertTokenizerFast,
    DistilBertTokenizer,
    ElectraTokenizerFast,
    RobertaTokenizerFast,
    DistilBertTokenizerFast,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from pytorch_widedeep.models.text.hf_utils import get_model_class
from pytorch_widedeep.utils.fastai_transforms import (
    fix_html,
    spec_add_spaces,
    rm_useless_spaces,
)

num_processes = os.cpu_count()


class HFTokenizer:
    def __init__(
        self,
        model_name: str,
        use_fast_tokenizer: bool = True,
        num_workers: Optional[int] = None,
        preprocessing_rules: Optional[List[Callable[[str], str]]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.use_fast_tokenizer = use_fast_tokenizer
        self.num_workers = num_workers
        self.preprocessing_rules = preprocessing_rules

        self._multiprocessing = num_workers is not None and num_workers > 1

        self.model_class = get_model_class(model_name)
        self.tokenizer = self._get_tokenizer(
            self.model_class, self.model_name, **kwargs
        )

    def fit(self, texts: List[str], **kwargs) -> "HFTokenizer":
        return self

    def transform(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        if self.preprocessing_rules:
            if self._multiprocessing:
                texts = self._process_text_parallel(texts)
            else:
                texts = [self._preprocess_text(text) for text in texts]

        if self.use_fast_tokenizer:
            encoded_texts = self.tokenizer.batch_encode_plus(
                texts,
                **kwargs,
            )
            input_ids = encoded_texts.get("input_ids")
        elif self._multiprocessing:
            input_ids = self._encode_paralell(texts, **kwargs)
        else:
            encoded_texts = self.tokenizer.batch_encode_plus(
                texts,
                **kwargs,
            )
            input_ids = encoded_texts.get("input_ids")

        return np.stack(input_ids)

    def fit_transform(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        return self.fit(texts, **kwargs).transform(texts, **kwargs)

    def inverse_transform(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens
    ) -> List[str]:
        texts = [
            self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens)
            )
            for i in range(input_ids.shape[0])
        ]
        return texts

    def encode(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        return self.transform(texts, **kwargs)

    def decode(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens: bool = False
    ) -> List[str]:
        return self.inverse_transform(input_ids, skip_special_tokens)

    def _get_tokenizer(
        self, model_class: str, model_name: str, **kwargs
    ) -> PreTrainedTokenizer:
        if model_class == "distilbert":
            return (
                DistilBertTokenizer.from_pretrained(model_name, **kwargs)
                if not self.use_fast_tokenizer
                else DistilBertTokenizerFast.from_pretrained(model_name, **kwargs)
            )

        if model_class == "bert":
            return (
                BertTokenizer.from_pretrained(model_name, **kwargs)
                if not self.use_fast_tokenizer
                else BertTokenizerFast.from_pretrained(model_name, **kwargs)
            )

        if model_class == "roberta":
            return (
                RobertaTokenizer.from_pretrained(model_name, **kwargs)
                if not self.use_fast_tokenizer
                else RobertaTokenizerFast.from_pretrained(model_name, **kwargs)
            )

        if model_class == "albert":
            return (
                AlbertTokenizer.from_pretrained(model_name, **kwargs)
                if not self.use_fast_tokenizer
                else AlbertTokenizerFast.from_pretrained(model_name, **kwargs)
            )

        if model_class == "electra":
            return (
                ElectraTokenizer.from_pretrained(model_name, **kwargs)
                if not self.use_fast_tokenizer
                else ElectraTokenizerFast.from_pretrained(model_name, **kwargs)
            )

    def _process_text_parallel(self, texts: List[str]) -> List[str]:
        num_processes = (
            self.num_workers if self.num_workers is not None else os.cpu_count()
        )
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            processed_texts = list(executor.map(self._preprocess_text, texts))
        return processed_texts

    def _preprocess_text(self, text: str) -> str:
        for rule in self.preprocessing_rules:
            text = rule(text)
        return text

    def _encode_paralell(self, texts: List[str], **kwargs) -> List[List[int]]:
        num_processes = (
            self.num_workers if self.num_workers is not None else os.cpu_count()
        )
        ptokenizer = partial(self.tokenizer.encode, **kwargs)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            encoded_texts = list(executor.map(ptokenizer, texts))
        return encoded_texts


if __name__ == "__main__":
    tokenizer = HFTokenizer(
        "distilbert-base-uncased",
        use_fast_tokenizer=False,
        num_workers=1,
        preprocessing_rules=[fix_html, spec_add_spaces, rm_useless_spaces],
    )

    texts = [
        "this is a test # 1",
        "this is another test       ",
        "this is a third test #39;",
    ]

    encoded_texts = tokenizer.fit_transform(
        texts,
        add_special_tokens=True,
        max_length=10,
        padding="max_length",
        truncation=True,
    )
