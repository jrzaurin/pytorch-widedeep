from importlib import resources

import pandas as pd


def load_bio_kdd04(as_frame: bool = False):
    """Load and return the higly imbalanced Protein Homology
    Dataset from [KDD cup 2004](https://www.kdd.org/kdd-cup/view/kdd-cup-2004/Data.
    This datasets include only bio_train.dat part of the dataset


    * The first element of each line is a BLOCK ID that denotes to which native sequence
    this example belongs. There is a unique BLOCK ID for each native sequence.
    BLOCK IDs are integers running from 1 to 303 (one for each native sequence,
    i.e. for each query). BLOCK IDs were assigned before the blocks were split
    into the train and test sets, so they do not run consecutively in either file.
    * The second element of each line is an EXAMPLE ID that uniquely describes
    the example. You will need this EXAMPLE ID and the BLOCK ID when you submit results.
    * The third element is the class of the example. Proteins that are homologous to
    the native sequence are denoted by 1, non-homologous proteins (i.e. decoys) by 0.
    Test examples have a "?" in this position.
    * All following elements are feature values. There are 74 feature values in each line.
    The features describe the match (e.g. the score of a sequence alignment) between
    the native protein sequence and the sequence that is tested for homology.
    """

    header_list = ["EXAMPLE_ID", "BLOCK_ID", "target"] + [str(i) for i in range(4, 78)]
    with resources.path("pytorch_widedeep.datasets.data", "bio_train.dat") as fpath:
        df = pd.read_csv(fpath, sep="\t", names=header_list)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_adult(as_frame: bool = False):
    """Load and return the [adult income datatest](http://www.cs.toronto.edu/~delve/data/adult/desc.html).
    you may find detailed description [here](http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html)
    """

    with resources.path("pytorch_widedeep.datasets.data", "adult.csv.zip") as fpath:
        df = pd.read_csv(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()
