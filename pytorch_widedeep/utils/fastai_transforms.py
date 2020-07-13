"""
NLP data processing; tokenizes text and creates vocab indexes

I have directly copied and paste part of OF THE TRANSFORMS.PY FASTAI LIBRARY.
I only need the Tokenizer and the Vocab classes which are both in this module.
This way I avoid the numerous fastai dependencies.

Credit for the code here to Jeremy Howard and the fastai team
"""

import os
import re
import html
from collections import Counter, defaultdict
from concurrent.futures.process import ProcessPoolExecutor

import spacy
from spacy.symbols import ORTH

from ..wdtypes import *


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
        return len(os.sched_getaffinity(0))
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
    """Basic class for a tokenizer function.
    """

    def __init__(self, lang: str):
        self.lang = lang

    def tokenizer(self, t: str) -> List[str]:
        return t.split(" ")

    def add_special_cases(self, toks: Collection[str]):
        pass


class SpacyTokenizer(BaseTokenizer):
    """Wrapper around a spacy tokenizer to make it a :obj:`BaseTokenizer`.

    Parameters
    ----------
    lang: str
        Language of the text to be tokenized
    """

    def __init__(self, lang: str):
        self.tok = spacy.blank(lang, disable=["parser", "tagger", "ner"])

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
        toks: Collection[str]
            `List`, `Tuple`, `Set` or `Dictionary` with special cases
            to add to the tokenizer
        """
        for w in toks:
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])


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
    """Class to combine a series of rules and a tokenizer function to tokenize
    text with multiprocessing.

    Parameters
    ----------
    tok_func: Callable, Default = SpacyTokenizer
        Tokenizer Object. See :class:`pytorch_widedeep.utils.fastai_transforms.SpacyTokenizer`
    lang: str, Default = "en",
        Text's Language
    pre_rules: ListRules, Default = None,
        Custom type, see :obj:`pytorch_widedeep.wdtypes`. Preprocessing Rules
    post_rules: ListRules, Default = None,
        Custom type, see :obj:`pytorch_widedeep.wdtypes`. Postprocessing Rules
    special_cases: Collection[str], Default= None,
        special cases to be added to the tokenizer via ``Spacy``'s
        ``add_special_case`` method
    n_cpus: int, Default = None
        number of CPUs to used during the tokenization process
    """

    def __init__(
        self,
        tok_func: Callable = SpacyTokenizer,
        lang: str = "en",
        pre_rules: ListRules = None,
        post_rules: ListRules = None,
        special_cases: Collection[str] = None,
        n_cpus: int = None,
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
        """Process and tokenize one text ``t`` with tokenizer ``tok``.

        Parameters
        ----------
        t: str
            text to be processed and tokenized
        tok: BaseTokenizer
            Instance of :obj:`BaseTokenizer`
        """
        for rule in self.pre_rules:
            t = rule(t)
        toks = tok.tokenizer(t)
        for rule in self.post_rules:
            toks = rule(toks)
        return toks

    def _process_all_1(self, texts: Collection[str]) -> List[List[str]]:
        """Process a list of ``texts`` in one process.
        """

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
            [['xxmaj', 'machine', 'learning', 'is', 'great'],
            ['but', 'building', 'stuff', 'is', 'even', 'better']]

        .. note:: Note the token ``TK_MAJ`` (`xxmaj`), used to indicate the
            next word begins with a capital in the original text. For more
            details of special tokens please see the ``fastai`` `docs
            <https://docs.fast.ai/text.transform.html#Tokenizer>`_.
        """

        if self.n_cpus <= 1:
            return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(
                e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), []
            )


class Vocab:
    """Contains the correspondence between numbers and tokens.

    Parameters
    ----------
    itos: Collection[str]
        `index to str`. Collection of srt that are the tokens of the vocabulary

    Attributes
    ----------
    stoi: defaultdict
        `str to index`. Dictionary containing the tokens of the vocabulary and
        their corresponding index
    """

    def __init__(self, itos: Collection[str]):
        self.itos = itos
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    def numericalize(self, t: Collection[str]) -> List[int]:
        """Convert a list of str (or tokens) ``t`` to their ids.
        """
        return [self.stoi[w] for w in t]

    def textify(self, nums: Collection[int], sep=" ") -> List[str]:
        """Convert a list of ``nums`` (or indexes) to their tokens.
        """
        return sep.join([self.itos[i] for i in nums]) if sep is not None else [self.itos[i] for i in nums]  # type: ignore

    def __getstate__(self):
        return {"itos": self.itos}

    def __setstate__(self, state: dict):
        self.itos = state["itos"]
        self.stoi = defaultdict(int, {v: k for k, v in enumerate(self.itos)})

    def save(self, path):
        """Save the  attribute ``self.itos`` in ``path``"""
        pickle.dump(self.itos, open(path, "wb"))

    @classmethod
    def create(cls, tokens: Tokens, max_vocab: int, min_freq: int) -> "Vocab":
        r"""Create a vocabulary object from a set of tokens.

        Parameters
        ----------
        tokens: Tokens
            Custom type, see :obj:`pytorch_widedeep.wdtypes`. Collection of
            collection of str (e.g. list of tokenized sentences)
        max_vocab: int
            maximum vocabulary size
        min_freq: int
            minimum frequency that a token has to appear to be part of the
            vocabulary

        Examples
        --------
        >>> from pytorch_widedeep.utils import Tokenizer, Vocab
        >>> texts = ['Machine learning is great', 'but building stuff is even better']
        >>> tokens = Tokenizer().process_all(texts)
        >>> vocab = Vocab.create(tokens, max_vocab=18, min_freq=1)
        >>> print(vocab.itos)
        ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep',
        'is', 'machine', 'learning', 'great', 'but', 'building', 'stuff', 'even', 'better']
        >>> vocab.numericalize(['machine', 'learning', 'is', 'great'])
        [10, 11, 9, 12]
        >>> vocab.textify([10, 11, 9, 12])
        'machine learning is great'

        .. note:: Note the many special tokens that ``fastai``'s' tokenizer
            adds. These are particularly useful when building Language models and/or in
            classification/Regression tasks. Please see the ``fastai``
            `docs <https://docs.fast.ai/text.transform.html#Tokenizer>`_.
        """
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o, c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(defaults.text_spec_tok):
            if o in itos:
                itos.remove(o)
            itos.insert(0, o)
        itos = itos[:max_vocab]
        if (
            len(itos) < max_vocab
        ):  # Make sure vocab size is a multiple of 8 for fast mixed precision training
            while len(itos) % 8 != 0:
                itos.append("xxfake")
        return cls(itos)

    @classmethod
    def load(cls, path):
        """Load an intance of :obj:`Vocab` contained in ``path``"""
        itos = pickle.load(open(path, "rb"))
        return cls(itos)
