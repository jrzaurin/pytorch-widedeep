import os
import warnings
from typing import Any, Dict, List, Callable, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import numpy.typing as npt

from pytorch_widedeep.utils.hf_utils import get_tokenizer
from pytorch_widedeep.preprocessing.base_preprocessor import BasePreprocessor

num_processes = os.cpu_count()


class HFPreprocessor(BasePreprocessor):
    def __init__(
        self,
        model_name: str,
        *,
        use_fast_tokenizer: bool = True,
        text_col: Optional[str] = None,
        num_workers: Optional[int] = None,
        preprocessing_rules: Optional[List[Callable[[str], str]]] = None,
        tokenizer_params: Optional[Dict[str, Any]] = None,
        encode_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.use_fast_tokenizer = use_fast_tokenizer
        self.text_col = text_col
        self.num_workers = num_workers
        self.preprocessing_rules = preprocessing_rules
        self.tokenizer_params = tokenizer_params if tokenizer_params is not None else {}
        self.encode_params = encode_params if encode_params is not None else {}

        self._multiprocessing = num_workers is not None and num_workers > 1

        self.tokenizer = get_tokenizer(
            self.model_name, **self.tokenizer_params, **kwargs
        )

        # A HuggingFace tokenizer is already trained, since we need this
        # attribute elsewhere in the library, we simply set it to True
        self.is_fitted = True

    def encode(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:

        if kwargs:
            self.encode_params.update(kwargs)

        if self.preprocessing_rules:
            if self._multiprocessing:
                texts = self._process_text_parallel(texts)
            else:
                texts = [self._preprocess_text(text) for text in texts]

        if self.use_fast_tokenizer:
            encoded_texts = self.tokenizer.batch_encode_plus(
                texts,
                **self.encode_params,
            )
            input_ids = encoded_texts.get("input_ids")
        elif self._multiprocessing:
            input_ids = self._encode_paralell(texts, **self.encode_params)
        else:
            encoded_texts = self.tokenizer.batch_encode_plus(
                texts,
                **self.encode_params,
            )
            input_ids = encoded_texts.get("input_ids")

        self.is_fitted = True

        try:
            output = np.array(input_ids)
        except ValueError:
            warnings.warn(
                "Padding and Truncating parameters were not passed and all input arrays "
                "do not have the same shape. Padding to the longest sequence. "
                "Padding will be done with the index of the pad token for the model",
                UserWarning,
            )
            max_len = max([len(ids) for ids in input_ids])
            output = np.array(
                [
                    np.pad(ids, (self.tokenizer.pad_token_id, max_len - len(ids)))
                    for ids in input_ids
                ]
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

    def fit(self, df: pd.DataFrame) -> "HFPreprocessor":
        # Note that this method is only included here for consistency with the
        # rest of the library and with the BasePreprocessor in particular.
        # HuggingFace's tokenizers and models are already trained. Therefore,
        # the 'fit' method here does nothing
        if self.text_col is None:
            raise ValueError(
                "'text_col' is None. Please specify the column name containing the text data"
                " if you want to use the 'fit' method"
            )
        return self

    def transform(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        if self.text_col is None:
            raise ValueError(
                "'text_col' is None. Please specify the column name containing the text data"
                " if you want to use the 'fit' method"
            )

        texts = self._read_texts(df)

        return self.encode(texts)

    def transform_sample(self, text: str) -> npt.NDArray[np.int64]:
        if not self.is_fitted:
            raise ValueError(
                "The `encode` (or `fit`) method must be called before calling `transform_sample`"
            )
        return self.encode([text])[0]

    def fit_transform(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        return self.fit(df).transform(df)

    def inverse_transform(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens: bool
    ) -> List[str]:
        return self.decode(input_ids, skip_special_tokens)

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

    def _read_texts(
        self, df: pd.DataFrame, root_dir: Optional[str] = None
    ) -> List[str]:
        if root_dir is not None:
            if not os.path.exists(root_dir):
                raise ValueError(
                    "root_dir does not exist. Please create it before fitting the preprocessor"
                )
            texts_fnames = df[self.text_col].tolist()
            texts: List[str] = []
            for texts_fname in texts_fnames:
                with open(os.path.join(root_dir, texts_fname), "r") as f:
                    texts.append(f.read().replace("\n", ""))
        else:
            texts = df[self.text_col].tolist()

        return texts

    def __repr__(self):
        return (
            f"HFPreprocessor(text_col={self.text_col}, model_name={self.model_name}, "
            f"use_fast_tokenizer={self.use_fast_tokenizer}, num_workers={self.num_workers}, "
            f"preprocessing_rules={self.preprocessing_rules}, tokenizer_params={self.tokenizer_params}, "
            f"encode_params={self.encode_params})"
        )


class ChunkHFPreprocessor(HFPreprocessor):
    def __init__(
        self,
        model_name: str,
        *,
        text_col: str,
        root_dir: Optional[str] = None,
        use_fast_tokenizer: bool = True,
        num_workers: Optional[int] = None,
        preprocessing_rules: Optional[List[Callable[[str], str]]] = None,
        tokenizer_params: Optional[Dict[str, Any]] = None,
        encode_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            model_name=model_name,
            use_fast_tokenizer=use_fast_tokenizer,
            text_col=text_col,
            num_workers=num_workers,
            preprocessing_rules=preprocessing_rules,
            tokenizer_params=tokenizer_params,
            encode_params=encode_params,
        )

        self.root_dir = root_dir

        # when using in chunks encode_params is not really optional. I will
        # review types in due time
        if self.encode_params is None:
            raise ValueError(
                "The 'encode_params' dict must be passed to the ChunkHFTokenizer "
                "containing at least the 'max_length' encoding parameter"
            )

        if "padding" not in self.encode_params or not self.encode_params["padding"]:
            self.encode_params["padding"] = True

        if (
            "truncation" not in self.encode_params
            or not self.encode_params["truncation"]
        ):
            self.encode_params["truncation"] = True

    def __repr__(self):
        return (
            f"ChunkHFPreprocessor(text_col={self.text_col}, model_name={self.model_name}, "
            f"use_fast_tokenizer={self.use_fast_tokenizer}, num_workers={self.num_workers}, "
            f"preprocessing_rules={self.preprocessing_rules}, tokenizer_params={self.tokenizer_params}, "
            f"encode_params={self.encode_params}, root_dir={self.root_dir})"
        )
