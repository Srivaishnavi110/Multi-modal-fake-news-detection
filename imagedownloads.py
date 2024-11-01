import os
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('fakeddit_sampled.csv')
def preprocess_dataset(df):
    if not os.path.exists("images"):
        os.makedirs("images")
    pbar = tqdm(total=len(df))
    for index, row in df.iterrows():
        if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
            try:
                image_url = row["image_url"]
                image_path = "images/" + str(row["id"]) + ".jpg"
                urllib.request.urlretrieve(image_url, image_path)
                df.at[index, "local_image_path"] = image_path
            except Exception as e:
                # If the image cannot be downloaded, set the local_image_path to NaN
                df.at[index, "local_image_path"] = np.nan
        else:
            df.at[index, "local_image_path"] = np.nan
        pbar.update(1)
    pbar.close()

    # Drop rows where the image could not be downloaded (i.e., local_image_path is NaN)
    df_cleaned = df.dropna(subset=["local_image_path"])
    return df_cleaned
print(preprocess_dataset(df))