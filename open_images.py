import numpy as np
import pandas as pd

import glob
import typing

def open_image_5d(file: str,
               image_id: int,
               channel_id: int,
               stack_id: int) -> np.array:
    return image

def open_image_4d(file: str,
               image_id: int,
               channel_id: int) -> np.array:
    return image

def registration(folder: str,
                 old_df: pd.DataFrame(columns=["path"])) -> pd.DataFrame:
    df = pd.concat(
        map(
            register_image,
            list(set(glob.glob(f"{folder}/**/*.nd2",recursive=True))
                 - set(old_df.path.unique()))
        )
    )
    return pd.concat(old_df,df)

def register_image(file: str) -> pd.DataFrame:
    return df 