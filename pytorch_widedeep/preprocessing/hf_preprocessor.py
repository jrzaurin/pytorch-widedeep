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
    """Text processor to prepare the ``deeptext`` input dataset that is a
    wrapper around HuggingFace's tokenizers.

    Following the main phylosophy of the `pytorch-widedeep` library, this
    class is designed to be as flexible as possible. Therefore, it is coded
    so that the user can use it as one would use any HuggingFace tokenizers,
    or following the API call 'protocol' of the rest of the library.

    Parameters
    ----------
    model_name: str
        The model name from the transformers library e.g. _'bert-base-uncased'_.
        Currently supported models are those from the families: BERT, RoBERTa,
        DistilBERT, ALBERT and ELECTRA.
    use_fast_tokenizer: bool, default = False
        Whether to use the fast tokenizer from HuggingFace or not
    text_col: Optional[str], default = None
        The column in the input dataframe containing the text data. If this
        tokenizer is used via the `fit` and `transform` methods, this
        argument is mandatory. If the tokenizer is used via the `encode`
        method, this argument is not needed since the input text is passed
        directly to the `encode` method.
    num_workers: Optional[int], default = None
        Number of workers to use when preprocessing the text data. If not
        None, and `use_fast_tokenizer` is False, the text data will be
        preprocessed in parallel using the number of workers specified. If
        `use_fast_tokenizer` is True, this argument is ignored.
    preprocessing_rules: Optional[List[Callable[[str], str]]], default = None
        A list of functions to be applied to the text data before encoding.
        This can be useful to clean the text data before encoding. For
        example, removing html tags, special characters, etc.
    tokenizer_params: Optional[Dict[str, Any]], default = None
        Additional parameters to be passed to the HuggingFace's
        `PreTrainedTokenizer`. Parameters to the `PreTrainedTokenizer`
        can also be passed via the `**kwargs` argument
    encode_params: Optional[Dict[str, Any]], default = None
        Additional parameters to be passed to the `batch_encode_plus` method
        of the HuggingFace's `PreTrainedTokenizer`. If the `fit` and `transform`
        methods are used, the `encode_params` dict parameter is mandatory. If
        the `encode` method is used, this parameter is not needed since the
        input text is passed directly to the `encode` method.
    **kwargs
        Additional kwargs to be passed to the model, in particular to the
        `PreTrainedTokenizer` class.

    Attributes
    ----------
    is_fitted: bool
        Boolean indicating if the preprocessor has been fitted. This is a
        HuggingFacea tokenizer, so it is always considered fitted and this
        attribute is manually set to True internally. This parameter exists
        for consistency with the rest of the library and because is needed
        for some functionality in the library.

    Examples
    --------
    >>> import pandas as pd
    >>> from pytorch_widedeep.preprocessing import HFPreprocessor
    >>> df = pd.DataFrame({"text": ["this is the first text", "this is the second text"]})
    >>> hf_processor_1 = HFPreprocessor(model_name="bert-base-uncased", text_col="text")
    >>> X_text_1 = hf_processor_1.fit_transform(df)
    >>> texts = ["this is a new text", "this is another text"]
    >>> hf_processor_2 = HFPreprocessor(model_name="bert-base-uncased")
    >>> X_text_2 = hf_processor_2.encode(texts, max_length=10, padding="max_length", truncation=True)
    """

    def __init__(
        self,
        model_name: str,
        *,
        use_fast_tokenizer: bool = False,
        text_col: Optional[str] = None,
        root_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
        preprocessing_rules: Optional[List[Callable[[str], str]]] = None,
        tokenizer_params: Optional[Dict[str, Any]] = None,
        encode_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.use_fast_tokenizer = use_fast_tokenizer
        self.text_col = text_col
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.preprocessing_rules = preprocessing_rules
        self.tokenizer_params = tokenizer_params if tokenizer_params is not None else {}
        self.encode_params = encode_params if encode_params is not None else {}

        self._multiprocessing = (
            num_workers is not None and num_workers > 1 and not use_fast_tokenizer
        )

        if kwargs:
            self.tokenizer_params.update(kwargs)

        self.tokenizer = get_tokenizer(
            model_name=self.model_name,
            use_fast_tokenizer=self.use_fast_tokenizer,
            **self.tokenizer_params,
        )

        # A HuggingFace tokenizer is already trained, since we need this
        # attribute elsewhere in the library, we simply set it to True
        self.is_fitted = True

    def encode(self, texts: List[str], **kwargs) -> npt.NDArray[np.int64]:
        """
        Encodes a list of texts. The method is a wrapper around the
        `batch_encode_plus` method of the HuggingFace's tokenizer.

        if 'use_fast_tokenizer' is True, the method will use the `batch_encode_plus`

        Parameters
        ----------
        texts: List[str]
            List of texts to be encoded
        **kwargs
            Additional parameters to be passed to the `batch_encode_plus` method
            of the HuggingFace's tokenizer. If the 'encode_params' dict was passed
            when instantiating the class, that dictionaly will be updated with
            the kwargs passed here.

        Returns
        -------
        np.array
            The encoded texts
        """
        if kwargs:
            self.encode_params.update(kwargs)

        if self.preprocessing_rules:
            if self._multiprocessing:
                texts = self._process_text_parallel(texts)
            else:
                texts = [self._preprocess_text(text) for text in texts]

        if self._multiprocessing:
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
        """
        Decodes a list of input_ids. The method is a wrapper around the
        `convert_ids_to_tokens` and `convert_tokens_to_string` methods of the
        HuggingFace's tokenizer.

        Parameters
        ----------
        input_ids: npt.NDArray[np.int64]
            The input_ids to be decoded
        skip_special_tokens: bool
            Whether to skip the special tokens or not

        Returns
        -------
        List[str]
            The decoded texts
        """
        texts = [
            self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens)
            )
            for i in range(input_ids.shape[0])
        ]
        return texts

    def fit(self, df: pd.DataFrame) -> "HFPreprocessor":
        """
        This method is included for consistency with the rest of the library
        in general and with the `BasePreprocessor` in particular. HuggingFace's
        tokenizers and models are already trained. Therefore, the 'fit' method
        here does nothing other than checking that the 'text_col' parameter is
        not `None`.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the text data in the column specified by
            the 'text_col' parameter
        """
        if self.text_col is None:
            raise ValueError(
                "'text_col' is None. Please specify the column name containing the text data"
                " if you want to use the 'fit' method"
            )
        return self

    def transform(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        """
        Encodes the text data in the input dataframe. This method simply
        calls the `encode` method under the hood. Similar to the `fit` method,
        this method is included for consistency with the rest of the library
        in general and with the `BasePreprocessor` in particular.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the text data in the column specified by
            the 'text_col' parameter

        Returns
        -------
        np.array
            The encoded texts
        """
        if self.text_col is None:
            raise ValueError(
                "'text_col' is None. Please specify the column name containing the text data"
                " if you want to use the 'fit' method"
            )

        texts = self._read_texts(df, self.root_dir)

        return self.encode(texts)

    def transform_sample(self, text: str) -> npt.NDArray[np.int64]:
        """
        Encodes a single text sample.

        Parameters
        ----------
        text: str
            The text sample to be encoded

        Returns
        -------
        np.array
            The encoded text
        """

        if not self.is_fitted:
            raise ValueError(
                "The `encode` (or `fit`) method must be called before calling `transform_sample`"
            )
        return self.encode([text])[0]

    def fit_transform(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        """
        Encodes the text data in the input dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the text data in the column specified by
            the 'text_col' parameter

        Returns
        -------
        np.array
            The encoded texts
        """
        return self.fit(df).transform(df)

    def inverse_transform(
        self, input_ids: npt.NDArray[np.int64], skip_special_tokens: bool
    ) -> List[str]:
        """
        Decodes a list of input_ids. The method simply calls the `decode` method
        under the hood.

        Parameters
        ----------
        input_ids: npt.NDArray[np.int64]
            The input_ids to be decoded
        skip_special_tokens: bool
            Whether to skip the special tokens or not

        Returns
        -------
        List[str]
            The decoded texts
        """
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
    """Text processor to prepare the ``deeptext`` input dataset that is a
    wrapper around HuggingFace's tokenizers.

    Hugginface Tokenizer's are already 'trained'. Therefore, unlike the
    `ChunkTextPreprocessor` this is mostly identical to the `HFPreprocessor`
    with the only difference that the class needs a 'text_col' parameter to
    be passed. Also the parameter `encode_params` is not really optional when
    using this class. It must be passed containing at least the
    'max_length' encoding parameter. This is because we need to ensure that
     all sequences have the same length when encoding in chunks.

    Parameters
    ----------
    model_name: str
        The model name from the transformers library e.g. _'bert-base-uncased'_.
        Currently supported models are those from the families: BERT, RoBERTa,
        DistilBERT, ALBERT and ELECTRA.
    text_col: str, default = None
        The column in the input dataframe containing the text data. When using
        the `ChunkHFPreprocessor` the `text_col` parameter is mandatory.
    root_dir: Optional[str], default = None
        The root directory where the text files are located. This is only
        needed if the text data is stored in text files. If the text data is
        stored in a column in the input dataframe, this parameter is not
        needed.
    use_fast_tokenizer: bool, default = False
        Whether to use the fast tokenizer from HuggingFace or not
    num_workers: Optional[int], default = None
        Number of workers to use when preprocessing the text data. If not
        None, and `use_fast_tokenizer` is False, the text data will be
        preprocessed in parallel using the number of workers specified. If
        `use_fast_tokenizer` is True, this argument is ignored.
    preprocessing_rules: Optional[List[Callable[[str], str]]], default = None
        A list of functions to be applied to the text data before encoding.
        This can be useful to clean the text data before encoding. For
        example, removing html tags, special characters, etc.
    tokenizer_params: Optional[Dict[str, Any]], default = None
        Additional parameters to be passed to the HuggingFace's
        `PreTrainedTokenizer`.
    encode_params: Optional[Dict[str, Any]], default = None
        Additional parameters to be passed to the `batch_encode_plus` method
        of the HuggingFace's `PreTrainedTokenizer`. In the case of the
        `ChunkHFPreprocessor`, this parameter is not really `Optional`. It
        must be passed containing at least the 'max_length' encoding
        parameter

    Attributes
    ----------
    is_fitted: bool
        Boolean indicating if the preprocessor has been fitted. This is a
        HuggingFacea tokenizer, so it is always considered fitted and this
        attribute is manually set to True internally. This parameter exists
        for consistency with the rest of the library and because is needed
        for some functionality in the library.

    """

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
