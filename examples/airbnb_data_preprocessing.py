# coding: utf-8
import os
import warnings
from pathlib import Path
from functools import reduce
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/airbnb")
fname = "listings.csv.gz"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

df_original = pd.read_csv(DATA_PATH / fname)
print(df_original.shape)
df_original.head()

# this is just subjective. One can choose some other columns
keep_cols = [
    "id",
    "host_id",
    "description",
    "house_rules",
    "host_name",
    "host_listings_count",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "is_location_exact",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "security_deposit",
    "cleaning_fee",
    "guests_included",
    "extra_people",
    "minimum_nights",
    "instant_bookable",
    "cancellation_policy",
    "reviews_per_month",
]

df = df_original[keep_cols]
df = df[~df.reviews_per_month.isna()]
df = df[~df.description.isna()]
df = df[~df.host_listings_count.isna()]
print(df.shape)

# This is a preprocessing stage before preparing the data to be passed to WideDeep
# Let's go "column by column"

# house rules
#
# I will simply include a binary column with 1/0 if the property has/has not
# house rules.
df["has_house_rules"] = df["house_rules"]
df.has_house_rules.fillna(0, inplace=True)
df["has_house_rules"][df.has_house_rules != 0] = 1
df.drop("house_rules", axis=1, inplace=True)

# host_name
#
# I will use names to infer gender using `gender_guesser`

host_name = df.host_name.tolist()
d = gender.Detector()
host_gender = [d.get_gender(n) for n in host_name]
replace_dict = {"mostly_male": "male", "mostly_female": "female", "andy": "unknown"}
host_gender = [replace_dict.get(item, item) for item in host_gender]
Counter(host_gender)
df["host_gender"] = host_gender
df.drop("host_name", axis=1, inplace=True)
df.head()

# property_type, room_type, accommodates, bathrooms, bedrooms, beds and
# guests_included, host_listings_count, minimum_nights
#
# Here some standard pre-processing
df.property_type.value_counts()
replace_prop_type = [
    val
    for val in df.property_type.unique().tolist()
    if val not in ["Apartment", "House"]
]
replace_prop_type = {k: "other" for k in replace_prop_type}
df.property_type.replace(replace_prop_type, inplace=True)
df["property_type"] = df.property_type.apply(lambda x: "_".join(x.split(" ")).lower())

df.room_type.value_counts()
df["room_type"] = df.room_type.apply(lambda x: "_".join(x.split(" ")).lower())

df["bathrooms"][(df.bathrooms.isna()) & (df.room_type == "private_room")] = 0
df["bathrooms"][(df.bathrooms.isna()) & (df.room_type == "entire_home/apt")] = 1
df.bedrooms.fillna(1, inplace=True)
df.beds.fillna(1, inplace=True)

# Encode some as categorical
categorical_cut = [
    ("accommodates", 3),
    ("guests_included", 3),
    ("minimum_nights", 3),
    ("host_listings_count", 3),
    ("bathrooms", 1.5),
    ("bedrooms", 3),
    ("beds", 3),
]

for col, cut in categorical_cut:
    new_colname = col + "_catg"
    df[new_colname] = df[col].apply(lambda x: cut if x >= cut else x)
    df[new_colname] = df[new_colname].round().astype(int)

# Amenities
#
# I will just add a number of dummy columns with 1/0 if the property has/has
# not that particular amenity
amenity_repls = (
    ('"', ""),
    ("{", ""),
    ("}", ""),
    (" / ", "_"),
    ("/", "_"),
    (" ", "_"),
    ("(s)", ""),
)

amenities_raw = df.amenities.str.lower().tolist()
amenities = [
    reduce(lambda a, kv: a.replace(*kv), amenity_repls, s).split(",")
    for s in amenities_raw
]

all_amenities = list(chain(*amenities))
all_amenities_count = Counter(all_amenities)
all_amenities_count

# having a look to the list we see that one amenity is empty and two are
# "translation missing:..."
keep_amenities = []
for k, v in all_amenities_count.items():
    if k and "missing" not in k:
        keep_amenities.append(k)

final_amenities = [
    [amenity for amenity in house_amenities if amenity in keep_amenities]
    for house_amenities in amenities
]

# some properties have no amenities aparently
final_amenities = [
    ["no amenities"] if not amenity else amenity for amenity in final_amenities
]
final_amenities = [
    ["amenity_" + amenity for amenity in amenities] for amenities in final_amenities
]

# let's build the dummy df
df_list_of_amenities = pd.DataFrame({"groups": final_amenities}, columns=["groups"])
s = df_list_of_amenities["groups"]

mlb = MultiLabelBinarizer()

df_amenities = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=df.index)

df.drop("amenities", axis=1, inplace=True)
df = pd.concat([df, df_amenities], axis=1)
df.head()

# Price, security_deposit, cleaning_fee, extra_people

money_columns = ["price", "security_deposit", "cleaning_fee", "extra_people"]
tmp_money_df = df[money_columns].fillna("$0")

money_repls = (("$", ""), (",", ""))
for col in money_columns:
    val_str = tmp_money_df[col].tolist()
    val_num = [
        float(st)
        for st in [
            reduce(lambda a, kv: a.replace(*kv), money_repls, s) for s in val_str
        ]
    ]
    tmp_money_df[col] = val_num

high_price, high_deposit, high_cleaning_fee, high_extra_people = 1000, 2000, 200, 100

high_price_count = (tmp_money_df.price >= high_price).sum()
high_deposit_count = (tmp_money_df.security_deposit >= high_deposit).sum()
high_cleaning_fee_count = (tmp_money_df.cleaning_fee >= high_cleaning_fee).sum()
high_extra_people_count = (tmp_money_df.extra_people >= high_extra_people).sum()

print("properties with very high price: {}".format(high_price_count))
print("properties with very high security deposit: {}".format(high_deposit_count))
print("properties with very high cleaning fee: {}".format(high_cleaning_fee_count))
print("properties with very high extra people cost: {}".format(high_extra_people_count))

# We will now just concat and we will drop high values later one
df.drop(money_columns, axis=1, inplace=True)
df = pd.concat([df, tmp_money_df], axis=1)
df = df[
    (df.price < high_price)
    & (df.price != 0)
    & (df.security_deposit < high_deposit)
    & (df.cleaning_fee < high_cleaning_fee)
    & (df.extra_people < high_extra_people)
]
df.head()
print(df.shape)

# let's make sure there are no nan left
has_nan = df.isnull().any(axis=0)
has_nan = [df.columns[i] for i in np.where(has_nan)[0]]
if not has_nan:
    print("no NaN, all OK")

# Computing a proxi for yield

# Yield is defined as price * occupancy rate. Occupancy rate can be calculated
# by multiplying ((reviews / review rate) * average length of stay), where
# review rate and average length of stay are normally taken as a factor based
# in some model.  For example, in the San Francisco model a review rate of 0.5
# is used to convert reviews to estimated bookings (i.e. we assume that only
# half of the guests will leave a review). An average length of stay of 3
# nights  multiplied by the estimated bookings over a period gives the
# occupancy rate. Therefore, in the expression I have used below, if you want
# to turn my implementation of 'yield' into a "proper" one under the San
# Francisco model assumptions simply multiply my yield by 6 (3 * (1/0.5)) or
# by 72 (3 * 2 * 12) if you prefer per year.

df["yield"] = (df["price"] + df["cleaning_fee"]) * (df["reviews_per_month"])
df.drop(["price", "cleaning_fee", "reviews_per_month"], axis=1, inplace=True)
# we will focus in cases with yield below 600 (we lose ~3% of the data).
# No real reason for this, simply removing some "outliers"
df = df[df["yield"] <= 600]
df.to_csv(DATA_PATH / "listings_processed.csv", index=False)
print("data preprocessed finished. Final shape: {}".format(df.shape))
