"""
NLP data processing; tokenizes text and creates vocab indexes

I have directly copied and paste part of OF THE TRANSFORMS.PY FASTAI LIBRARY.
I only need the Tokenizer and the Vocab classes which are both in this module.
This way I avoid extra dependencies.

Credit for the code here to Jeremy Howard and the fastai team
"""

import os
import re
import html
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import spacy
from spacy.symbols import ORTH

from pytorch_widedeep.wdtypes import (
    Any,
    List,
    Match,
    Union,
    Tokens,
    Callable,
    Optional,
    ListRules,
    Collection,
    SimpleNamespace,
)


def partition(a: Collection, sz: int) -> List[Collection]:
    "Split iterables `a` in equal parts of size `sz`"
    return [a[i : i + sz] for i in range(0, len(a), sz)]  # type: ignore


def partition_by_cores(a: Collection, n_cpus: int) -> List[Collection]:
    "Split data in `a` equally among `n_cpus` cores"
    return partition(a, len(a) // n_cpus + 1)


def ifnone(a: Any, b: Any) -> Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


def num_cpus() -> Optional[int]:
    "Get number of cpus"
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count()


_default_cpus = min(16, num_cpus())
defaults = SimpleNamespace(
    cpus=_default_cpus, cmap="viridis", return_fig=False, silent=False
)

__all__ = [
    "BaseTokenizer",
    "SpacyTokenizer",
    "Tokenizer",
    "Vocab",
    "fix_html",
    "replace_all_caps",
    "replace_rep",
    "replace_wrep",
    "rm_useless_spaces",
    "spec_add_spaces",
    "BOS",
    "EOS",
    "FLD",
    "UNK",
    "PAD",
    "TK_MAJ",
    "TK_UP",
    "TK_REP",
    "TK_REP",
    "TK_WREP",
    "deal_caps",
]

BOS, EOS, FLD, UNK, PAD = "xxbos", "xxeos", "xxfld", "xxunk", "xxpad"
TK_MAJ, TK_UP, TK_REP, TK_WREP = "xxmaj", "xxup", "xxrep", "xxwrep"
defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_MAJ, TK_UP, TK_REP, TK_WREP]


class BaseTokenizer:
    """Basic class for a tokenizer function."""

    def __init__(self, lang: str):
        self.lang = lang

    def tokenizer(self, t: str) -> List[str]:
        return t.split(" ")

    def add_special_cases(self, toks: Collection[str]):
        pass


class SpacyTokenizer(BaseTokenizer):
    def __init__(self, lang: str):
        """Wrapper around a spacy tokenizer to make it a :obj:`BaseTokenizer`.

        Parameters
        ----------
        lang: str
            Language of the text to be tokenized
        """
        self.tok = spacy.blank(lang)

    def tokenizer(self, t: str):
        """Runs ``Spacy``'s ``tokenizer``

        Parameters
        ----------
        t: str
            text to be tokenized
        """
        return [t.text for t in self.tok.tokenizer(t)]

    def add_special_cases(self, toks: Collection[str]):
        """Runs ``Spacy``'s ``add_special_case`` method

        Parameters
        ----------
        toks: Collection
            `List`, `Tuple`, `Set` or `Dictionary` where the values are
            strings that are the special cases to add to the tokenizer
        """
        for w in toks:
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])  # type: ignore[union-attr]


def spec_add_spaces(t: str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)


def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)


def replace_rep(t: str) -> str:
    "Replace repetitions at the character level in `t`."

    def _replace_rep(m: Match[str]) -> str:
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t: str) -> str:
    "Replace word repetitions in `t`."

    def _replace_wrep(m: Match[str]) -> str:
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    re_wrep = re.compile(r"(\b\w+\W+)(\1{3,})")
    return re_wrep.sub(_replace_wrep, t)


def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace("<unk>", UNK)
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))


def replace_all_caps(x: Collection[str]) -> Collection[str]:
    "Replace tokens in ALL CAPS in `x` by their lower version and add `TK_UP` before."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1:
            res.append(TK_UP)
            res.append(t.lower())
        else:
            res.append(t)
    return res


def deal_caps(x: Collection[str]) -> Collection[str]:
    "Replace all Capitalized tokens in `x` by their lower version and add `TK_MAJ` before."
    res = []
    for t in x:
        if t == "":
            continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower():
            res.append(TK_MAJ)
        res.append(t.lower())
    return res


defaults.text_pre_rules = [
    fix_html,
    replace_rep,
    replace_wrep,
    spec_add_spaces,
    rm_useless_spaces,
]
defaults.text_post_rules = [replace_all_caps, deal_caps]


class Tokenizer:
    r"""Class to combine a series of rules and a tokenizer function to tokenize
    text with multiprocessing.

    Setting some of the parameters of this class require perhaps some
    familiarity with the source code.

    Parameters
    ----------
    tok_func: Callable, default = ``SpacyTokenizer``
        Tokenizer Object. See `pytorch_widedeep.utils.fastai_transforms.SpacyTokenizer`
    lang: str, default = "en"
        Text's Language
    pre_rules: ListRules, Optional, default = None
        Custom type: ``Collection[Callable[[str], str]]``. These are
        `Callable` objects that will be applied to the text (str) directly as
        `rule(tok)` before being tokenized.
    post_rules: ListRules, Optional, default = None
        Custom type: ``Collection[Callable[[str], str]]``. These are
        `Callable` objects that will be applied to the tokens as
        `rule(tokens)` after the text has been tokenized.
    special_cases: Collection, Optional, default= None
        special cases to be added to the tokenizer via ``Spacy``'s
        ``add_special_case`` method
    n_cpus: int, Optional, default = None
        number of CPUs to used during the tokenization process
    """

    def __init__(
        self,
        tok_func: Callable = SpacyTokenizer,
        lang: str = "en",
        pre_rules: Optional[ListRules] = None,
        post_rules: Optional[ListRules] = None,
        special_cases: Optional[Collection[str]] = None,
        n_cpus: Optional[int] = None,
    ):
        self.tok_func, self.lang, self.special_cases = tok_func, lang, special_cases
        self.pre_rules = ifnone(pre_rules, defaults.text_pre_rules)
        self.post_rules = ifnone(post_rules, defaults.text_post_rules)
        self.special_cases = (
            special_cases if special_cases is not None else defaults.text_spec_tok
        )
        self.n_cpus = ifnone(n_cpus, defaults.cpus)

    def __repr__(self) -> str:
        res = f"Tokenizer {self.tok_func.__name__} in {self.lang} with the following rules:\n"
        for rule in self.pre_rules:
            res += f" - {rule.__name__}\n"
        for rule in self.post_rules:
            res += f" - {rule.__name__}\n"
        return res

    def process_text(self, t: str, tok: BaseTokenizer) -> List[str]:
        r"""Process and tokenize one text ``t`` with tokenizer ``tok``.

        Parameters
        ----------
        t: str
            text to be processed and tokenized
        tok: ``BaseTokenizer``
            Instance of `BaseTokenizer`. See
            `pytorch_widedeep.utils.fastai_transforms.BaseTokenizer`

        Returns
        -------
        List[str]
            List of tokens
        """
        for rule in self.pre_rules:
            t = rule(t)
        toks = tok.tokenizer(t)
        for rule in self.post_rules:
            toks = rule(toks)
        return toks

    def _process_all_1(self, texts: Collection[str]) -> List[List[str]]:
        """Process a list of ``texts`` in one process."""

        tok = self.tok_func(self.lang)
        if self.special_cases:
            tok.add_special_cases(self.special_cases)
        return [self.process_text(str(t), tok) for t in texts]

    def process_all(self, texts: Collection[str]) -> List[List[str]]:
        r"""Process a list of texts. Parallel execution of ``process_text``.

        Examples
        --------
        >>> from pytorch_widedeep.utils import Tokenizer
        >>> texts = ['Machine learning is great', 'but building stuff is even better']
        >>> tok = Tokenizer()
        >>> tok.process_all(texts)
        [['xxmaj', 'machine', 'learning', 'is', 'great'], ['but', 'building', 'stuff', 'is', 'even', 'better']]

        :information_source: **NOTE**:
        Note the token ``TK_MAJ`` (`xxmaj`), used to indicate the
        next word begins with a capital in the original text. For more
        details of special tokens please see the [``fastai`` docs](https://docs.fast.ai/text.core.html#Tokenizing).

        Returns
        -------
        List[List[str]]
            List containing lists of tokens. One list per "_document_"

        """

        if self.n_cpus <= 1:
            return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(
                e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), []
            )


class Vocab:
    r"""Contains the correspondence between numbers and tokens.

    Parameters
    ----------
    max_vocab: int
        maximum vocabulary size
    min_freq: int
        minimum frequency for a token to be considereds
    pad_idx: int, Optional, default = None
        padding index. If `None`, Fastai's Tokenizer leaves the 0 index
        for the unknown token (_'xxunk'_) and defaults to 1 for the padding
        token (_'xxpad'_).

    Attributes
    ----------
    itos: Collection
        `index to str`. Collection of strings that are the tokens of the
        vocabulary
    stoi: defaultdict
        `str to index`. Dictionary containing the tokens of the vocabulary and
        their corresponding index
    """

    def __init__(
        self,
        max_vocab: int,
        min_freq: int,
        pad_idx: Optional[int] = None,
    ):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.pad_idx = pad_idx

    def create(
        self,
        tokens: Tokens,
    ) -> "Vocab":
        r"""Create a vocabulary object from a set of tokens.

        Parameters
        ----------
        tokens: Tokens
            Custom type: ``Collection[Collection[str]]``  see
            `pytorch_widedeep.wdtypes`. Collection of collection of
            strings (e.g. list of tokenized sentences)

        Examples
        --------
        >>> from pytorch_widedeep.utils import Tokenizer, Vocab
        >>> texts = ['Machine learning is great', 'but building stuff is even better']
        >>> tokens = Tokenizer().process_all(texts)
        >>> vocab = Vocab(max_vocab=18, min_freq=1).create(tokens)
        >>> vocab.numericalize(['machine', 'learning', 'is', 'great'])
        [10, 11, 9, 12]
        >>> vocab.textify([10, 11, 9, 12])
        'machine learning is great'

        :information_source: **NOTE**:
        Note the many special tokens that ``fastai``'s' tokenizer adds. These
        are particularly useful when building Language models and/or in
        classification/Regression tasks. Please see the [``fastai`` docs](https://docs.fast.ai/text.core.html#Tokenizing).

        Returns
        -------
        Vocab
            An instance of a `Vocab` object
        """

        freq = Counter(p for o in tokens for p in o)
        itos = [o for o, c in freq.most_common(self.max_vocab) if c >= self.min_freq]
        for o in reversed(defaults.text_spec_tok):
            if o in itos:
                itos.remove(o)
            itos.insert(0, o)

        if self.pad_idx is not None and self.pad_idx != 1:
            itos.remove(PAD)
            itos.insert(self.pad_idx, PAD)
            # get the new 'xxunk' index
            xxunk_idx = np.where([el == "xxunk" for el in itos])[0][0]
        else:
            xxunk_idx = 0

        itos = itos[: self.max_vocab]
        if (
            len(itos) < self.max_vocab
        ):  # Make sure vocab size is a multiple of 8 for fast mixed precision training
            while len(itos) % 8 != 0:
                itos.append("xxfake")

        self.itos = itos
        self.stoi = defaultdict(
            lambda: xxunk_idx, {v: k for k, v in enumerate(self.itos)}
        )

        return self

    def fit(
        self,
        tokens: Tokens,
    ) -> "Vocab":
        """
        Calls the `create` method. I simply want to honor fast ai naming, but
        for consistency with the rest of the library I am including a fit method
        """
        return self.create(tokens)

    def numericalize(self, t: Collection[str]) -> List[int]:
        """Convert a list of tokens ``t`` to their ids.

        Returns
        -------
        List[int]
            List of '_numericalsed_' tokens
        """
        return [self.stoi[w] for w in t]

    def transform(self, t: Collection[str]) -> List[int]:
        """
        Calls the `numericalize` method. I simply want to honor fast ai naming,
        but for consistency with the rest of the library I am including a
        transform method
        """
        return self.numericalize(t)

    def textify(self, nums: Collection[int], sep=" ") -> Union[str, List[str]]:
        """Convert a list of ``nums`` (or indexes) to their tokens.

        Returns
        -------
        List[str]
            List of tokens
        """
        return (
            sep.join([self.itos[i] for i in nums])
            if sep is not None
            else [self.itos[i] for i in nums]
        )

    def inverse_transform(
        self, nums: Collection[int], sep=" "
    ) -> Union[str, List[str]]:
        """
        Calls the `textify` method. I simply want to honor fast ai naming, but
        for consistency with the rest of the library I am including an
        inverse_transform method
        """
        # I simply want to honor fast ai naming, but for consistency with the
        # rest of the library I am including an inverse_transform method
        return self.textify(nums, sep)

    def __getstate__(self):
        return {"itos": self.itos}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})


class ChunkVocab:
    def __init__(
        self,
        max_vocab: int,
        min_freq: int,
        n_chunks: int,
        pad_idx: Optional[int] = None,
    ):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.n_chunks = n_chunks
        self.pad_idx = pad_idx

        self.chunk_counter = 0

        self.is_fitted = False

    def fit(
        self,
        tokens: Tokens,
    ) -> "ChunkVocab":
        if self.chunk_counter == 0:
            self.freq = Counter(tok for sent in tokens for tok in sent)
        else:
            self.freq.update(tok for sent in tokens for tok in sent)
        self.chunk_counter += 1

        if self.chunk_counter == self.n_chunks:
            itos = [
                o
                for o, c in self.freq.most_common(self.max_vocab)
                if c >= self.min_freq
            ]
            for o in reversed(defaults.text_spec_tok):
                if o in itos:
                    itos.remove(o)
                itos.insert(0, o)

            if self.pad_idx is not None:
                itos.remove(PAD)
                itos.insert(self.pad_idx, PAD)

            # get the new 'xxunk' index
            xxunk_idx = np.where([el == "xxunk" for el in itos])[0][0]

            itos = itos[: self.max_vocab]
            if (
                len(itos) < self.max_vocab
            ):  # Make sure vocab size is a multiple of 8 for fast mixed precision training
                while len(itos) % 8 != 0:
                    itos.append("xxfake")

            self.itos = itos
            self.stoi = defaultdict(
                lambda: xxunk_idx, {v: k for k, v in enumerate(self.itos)}
            )

            self.is_fitted = True

        return self

    def transform(self, t: Collection[str]) -> List[int]:
        """Convert a list of tokens ``t`` to their ids.

        Returns
        -------
        List[int]
            List of '_numericalsed_' tokens
        """
        return [self.stoi[w] for w in t]

    def inverse_transform(
        self, nums: Collection[int], sep=" "
    ) -> Union[str, List[str]]:
        """Convert a list of ``nums`` (or indexes) to their tokens.

        Returns
        -------
        List[str]
            List of tokens
        """
        return (
            sep.join([self.itos[i] for i in nums])
            if sep is not None
            else [self.itos[i] for i in nums]
        )

    def __getstate__(self):
        return {"itos": self.itos}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})
