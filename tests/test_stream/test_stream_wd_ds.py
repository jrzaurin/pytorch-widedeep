import os
import pandas as pd

from pytorch_widedeep.stream import StreamWideDeepDataset, StreamTextPreprocessor

path = './tests/test_stream/test.csv'

def test_iter_targetonly():
    X = pd.DataFrame([
        {'target': 0, 'text': 'python is cool'},
        {'target': 1, 'text': 'but rust might be cooler'}
    ])

    X.to_csv(path)

    try:
        ds = StreamWideDeepDataset(path, 'target')
        assert [x[1] for x in iter(ds)] == [0, 1]
    
    finally:
        os.remove(path)


def test_iter_text():
    X = pd.DataFrame([
        {'target': 0, 'text': 'python is cool'},
        {'target': 1, 'text': 'but rust might be cooler'}
    ])

    X.to_csv(path)

    try:
        proc = StreamTextPreprocessor('text', min_freq=1, pad_first=False)
        proc.fit(path, 2)
        ds = StreamWideDeepDataset(path, 'target', text_col='text', text_preprocessor=proc)
        i = [(x, target) for x, target in iter(ds)]
        print(i)
        assert len(i) == 2
        assert len(i[0]) == 80  # Padding is happening to 80 with inverse transform - is this a bug?
        assert len(i[1]) == 80
    
    finally:
        os.remove(path)

