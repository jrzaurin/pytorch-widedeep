import os
import warnings
from typing import List, Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt

from pytorch_widedeep.models.text.huggingface_transformers.hf_utils import (
    get_tokenizer,
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

        self.tokenizer = get_tokenizer(self.model_name, **kwargs)

        self.is_fitted = False

    def encode(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
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

        self.encoding_kwargs = kwargs
        self.is_fitted = True

        try:
            output = np.array(input_ids)
        except ValueError:
            warnings.warn(
                "Padding and Truncating parameters were not passed and all input arrays "
                "do not have the same shape. Padding to the longest sequence. "
                "Padding will be done with 0s added to the end of the sequences.",
                UserWarning,
            )
            max_len = max([len(ids) for ids in input_ids])
            output = np.array(
                [np.pad(ids, (0, max_len - len(ids))) for ids in input_ids]
            )

        return output

    def decode(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens: bool
    ) -> List[str]:
        texts = [
            self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens)
            )
            for i in range(input_ids.shape[0])
        ]
        return texts

    def fit(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        # this method exists only for consistency with the rest of the
        # classes
        return self.encode(texts, **kwargs)

    def transform(self, texts: List[str]) -> npt.NDArray[np.int64]:
        # this method exists only for consistency with the rest of the
        # classes
        if not self.is_fitted:
            raise ValueError(
                "The `encode` (or `fit`) method must be called before calling `transform`"
            )

        return self.encode(texts, **self.encoding_kwargs)

    def inverse_transform(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens: bool
    ) -> List[str]:
        return self.decode(input_ids, skip_special_tokens)

    def fit_transform(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        # this method exists only for consistency with the rest of the
        # classes
        return self.encode(texts, **kwargs)

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
