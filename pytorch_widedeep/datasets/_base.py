# dataframes are saved as parquet, pyarrow, brotli
# pd.to_parquet(path=None, engine="auto", compression="brotli", index=False)
# see related post: https://python.plainenglish.io/storing-pandas-98-faster-disk-reads-and-72-less-space-208e2e2be8bb
from importlib import resources

import pandas as pd


def load_bio_kdd04(as_frame: bool = False):
    """Load and return the higly imbalanced binary classification Protein Homology
    Dataset from [KDD cup 2004](https://www.kdd.org/kdd-cup/view/kdd-cup-2004/Data).
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

    # header_list = ["EXAMPLE_ID", "BLOCK_ID", "target"] + [str(i) for i in range(4, 78)]
    with resources.path(
        "pytorch_widedeep.datasets.data", "bio_train.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_adult(as_frame: bool = False):
    """Load and return the higly imbalanced binary classification [adult income datatest](http://www.cs.toronto.edu/~delve/data/adult/desc.html).
    you may find detailed description [here](http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html)
    """

    with resources.path(
        "pytorch_widedeep.datasets.data", "adult.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_ecoli(as_frame: bool = False):
    """Load and return the higly imbalanced multiclass classification e.coli dataset
    Dataset from [UCI Machine learning Repository](https://archive.ics.uci.edu/ml/datasets/ecoli).


    1. Title: Protein Localization Sites

    2. Creator and Maintainer:
            Kenta Nakai
                Institue of Molecular and Cellular Biology
            Osaka, University
            1-3 Yamada-oka, Suita 565 Japan
            nakai@imcb.osaka-u.ac.jp
                http://www.imcb.osaka-u.ac.jp/nakai/psort.html
    Donor: Paul Horton (paulh@cs.berkeley.edu)
    Date:  September, 1996
    See also: yeast database

    3. Past Usage.
    Reference: "A Probablistic Classification System for Predicting the Cellular
            Localization Sites of Proteins", Paul Horton & Kenta Nakai,
            Intelligent Systems in Molecular Biology, 109-115.
        St. Louis, USA 1996.
    Results: 81% for E.coli with an ad hoc structured
        probability model. Also similar accuracy for Binary Decision Tree and
        Bayesian Classifier methods applied by the same authors in
        unpublished results.

    Predicted Attribute: Localization site of protein. ( non-numeric ).

    4. The references below describe a predecessor to this dataset and its
    development. They also give results (not cross-validated) for classification
    by a rule-based expert system with that version of the dataset.

    Reference: "Expert Sytem for Predicting Protein Localization Sites in
            Gram-Negative Bacteria", Kenta Nakai & Minoru Kanehisa,
            PROTEINS: Structure, Function, and Genetics 11:95-110, 1991.

    Reference: "A Knowledge Base for Predicting Protein Localization Sites in
        Eukaryotic Cells", Kenta Nakai & Minoru Kanehisa,
        Genomics 14:897-911, 1992.

    5. Number of Instances:  336 for the E.coli dataset and

    6. Number of Attributes.
            for E.coli dataset:  8 ( 7 predictive, 1 name )

    7. Attribute Information.

    1.  Sequence Name: Accession number for the SWISS-PROT database
    2.  mcg: McGeoch's method for signal sequence recognition.
    3.  gvh: von Heijne's method for signal sequence recognition.
    4.  lip: von Heijne's Signal Peptidase II consensus sequence score.
            Binary attribute.
    5.  chg: Presence of charge on N-terminus of predicted lipoproteins.
        Binary attribute.
    6.  aac: score of discriminant analysis of the amino acid content of
        outer membrane and periplasmic proteins.
    7. alm1: score of the ALOM membrane spanning region prediction program.
    8. alm2: score of ALOM program after excluding putative cleavable signal
        regions from the sequence.

    8. Missing Attribute Values: None.

    9. Class Distribution. The class is the localization site. Please see Nakai & Kanehisa referenced above for more details.

    cp  (cytoplasm)                                    143
    im  (inner membrane without signal sequence)        77
    pp  (perisplasm)                                    52
    imU (inner membrane, uncleavable signal sequence)   35
    om  (outer membrane)                                20
    omL (outer membrane lipoprotein)                     5
    imL (inner membrane lipoprotein)                     2
    imS (inner membrane, cleavable signal sequence)      2
    """

    with resources.path(
        "pytorch_widedeep.datasets.data", "ecoli.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_california_housing(as_frame: bool = False):
    """Load and return the higly imbalanced regression California housing dataset.

    Characteristics:
    Number of Instances: 20640
    Number of Attributes: 8 numeric, predictive attributes and the target
    Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).

    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).

    An household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surpinsingly large values for block groups with few households
    and many empty houses, such as vacation resorts.

    References
    ----------
    Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
    Statistics and Probability Letters, 33 (1997) 291-297.
    """
    with resources.path(
        "pytorch_widedeep.datasets.data", "california_housing.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_birds(as_frame: bool = False):
    """Load and return the multi-label classification bird dataset.

    References
    ----------
    http://mulan.sourceforge.net/datasets-mlc.html

    F. Briggs, Yonghong Huang, R. Raich, K. Eftaxias, Zhong Lei, W. Cukierski, S. Hadley, A. Hadley,
    M. Betts, X. Fern, J. Irvine, L. Neal, A. Thomas, G. Fodor, G. Tsoumakas, Hong Wei Ng,
    Thi Ngoc Tho Nguyen, H. Huttunen, P. Ruusuvuori, T. Manninen, A. Diment, T. Virtanen,
    J. Marzat, J. Defretin, D. Callender, C. Hurlburt, K. Larrey, M. Milakov.
    "The 9th annual MLSP competition: New methods for acoustic classification of multiple
    simultaneous bird species in a noisy environment", in proc. 2013 IEEE International Workshop
    on Machine Learning for Signal Processing (MLSP)
    """
    with resources.path(
        "pytorch_widedeep.datasets.data", "birds_train.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_rf1(as_frame: bool = False):
    """Load and return the multi-target regression River Flow(RF1) dataset.

        Characterisctics:
        The river flow data set (RF1) concerns a prediction task in which flows in a river network are
    predicted for 48 hours in the future at 8 different locations in the Mississippi River network
    in the United States [18]. RF1 is one of the multi-target regression problems listed in the
    literature survey on multi-target regression problems by Borchani et al. [2], and therefore
    serves as a good test case for the active learning algorithm. Each row includes the most recent
    observation for each of the 8 sites as well as time-lagged observations from 6, 12, 18, 24, 36,
    48 and 60 hours in the past. Therefore, the data set consists in total of 64 attribute variables
    and 8 target variables. The data set contains over 1 year of hourly observations (over 9000
    data points) collected from September 2011 to September 2012 by the US National Weather
    Service. From these 9000 data points, 1000 points have been randomly sampled for training
    and 2000 for evaluation.
    """
    with resources.path(
        "pytorch_widedeep.datasets.data", "rf1_train.parquet.brotli"
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()


def load_womens_ecommerce(as_frame: bool = False):
    """
    Context
    This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers.
    Its nine supportive features offer a great environment to parse out the text through its multiple
    dimensions. Because this is real commercial data, it has been anonymized, and references to the company
    in the review text and body have been replaced with “retailer”.

    Content
    This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review,
    and includes the variables:

    Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
    Age: Positive Integer variable of the reviewers age.
    Title: String variable for the title of the review.
    Review Text: String variable for the review body.
    Rating: Positive Ordinal Integer variable for the product score granted by the customer from
        1 Worst, to 5 Best.
    Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended,
        0 is not recommended.
    Positive Feedback Count: Positive Integer documenting the number of other customers who found this
        review positive.
    Division Name: Categorical name of the product high level division.
    Department Name: Categorical name of the product department name.
    Class Name: Categorical name of the product class name.
    """
    with resources.path(
        "pytorch_widedeep.datasets.data",
        "WomensClothingE-CommerceReviews.parquet.brotli",
    ) as fpath:
        df = pd.read_parquet(fpath)

    if as_frame:
        return df
    else:
        return df.to_numpy()
