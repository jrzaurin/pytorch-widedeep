import os
import pickle
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from tqdm import tqdm


def download_images(df, out_path, id_col, img_col):
    download_error = []
    counter = 0
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if counter < 1000:
            img_path = str(out_path / ".".join([str(row[id_col]), "jpg"]))
            if os.path.isfile(img_path):
                continue
            else:
                try:
                    urlretrieve(row[img_col], img_path)
                    counter += 1
                except:
                    # print("Error downloading host image {}".format(row[id_col]))
                    download_error.append(row[id_col])
                    pass
    pickle.dump(download_error, open(DATA_PATH / (id_col + "_download_error.p"), "wb"))


if __name__ == "__main__":
    DATA_PATH = Path("data/airbnb")
    HOST_PATH = DATA_PATH / "host_picture"
    PROP_PATH = DATA_PATH / "property_picture"

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(HOST_PATH):
        os.makedirs(HOST_PATH)
    if not os.path.exists(PROP_PATH):
        os.makedirs(PROP_PATH)

    df_original = pd.read_csv(DATA_PATH / "listings.csv.gz")[
        ["id", "host_id", "picture_url", "host_picture_url"]
    ]
    df_processed = pd.read_csv(DATA_PATH / "listings_processed.csv")[["id", "host_id"]]
    df = df_processed.merge(df_original, on=["id", "host_id"])

    df_host = df.groupby("host_id").first().reset_index()
    download_images(df_host, HOST_PATH, id_col="host_id", img_col="host_picture_url")
    download_images(df, PROP_PATH, id_col="id", img_col="picture_url")
