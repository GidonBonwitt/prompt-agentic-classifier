import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def prepare_datasets(
    benign_path: str = "data/benign_prompts.parquet",
    harmful_path: str = "data/harmful_prompts.parquet",
    train_frac: float = 0.7,
    val_frac: float   = 0.15,
    test_frac: float  = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load benign and harmful prompts, label them, and split into 
    train/validation/test sets with stratification.

    Parameters
    ----------
    benign_path : str
        Path to the benign prompts Parquet file.
    harmful_path : str
        Path to the harmful prompts Parquet file.
    train_frac : float
        Fraction of data for training (default 0.7).
    val_frac : float
        Fraction for validation (default 0.15).
    test_frac : float
        Fraction for test (default 0.15).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_df, val_df, test_df : Tuple of pd.DataFrame
    """

    # 1. Load
    benign_df  = pd.read_parquet(benign_path)
    harmful_df = pd.read_parquet(harmful_path)

    # 2. Label
    benign_df  = benign_df.copy()
    harmful_df = harmful_df.copy()
    benign_df['label']  = 0
    harmful_df['label'] = 1

    # 3. Concat and shuffle
    full_df = pd.concat([benign_df, harmful_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 4. First split off the train set
    train_df, temp_df = train_test_split(
        full_df,
        test_size=(1.0 - train_frac),
        stratify=full_df['label'],
        random_state=random_state
    )

    # 5. Then split temp into val and test
    #    test_frac relative to temp_df is test_frac/(val_frac+test_frac)
    rel_test_size = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test_size,
        stratify=temp_df['label'],
        random_state=random_state
    )

    # 6. Reset indices
    for split in (train_df, val_df, test_df):
        split.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df


import pandas as pd

def load_batch(
    df: pd.DataFrame,
    n_samples: int = 2
) -> pd.DataFrame:
    """
    Load a batch of prompts with a balanced sample of benign and harmful examples.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing either a 'label' column (0=benign, 1=harmful)
        or a 'jailbreak' boolean column (False=benign, True=harmful).
    n_samples : int
        Number of examples to draw *per class*.

    Returns
    -------
    pd.DataFrame
        A shuffled DataFrame of size up to 2*n_samples, balanced between
        benign and harmful prompts.
    """
    # Ensure we have a numeric label column
    data = df.copy()
    if 'label' in data.columns:
        pass
    elif 'jailbreak' in data.columns:
        data['label'] = data['jailbreak'].astype(int)
    else:
        raise ValueError("DataFrame must contain either 'label' or 'jailbreak' column.")

    # Split by class
    benign_df  = data[data['label'] == 0]
    harmful_df = data[data['label'] == 1]

    # Sample up to n_samples from each (without replacement)
    benign_batch  = benign_df.sample(n=min(n_samples, len(benign_df)), random_state=42)
    harmful_batch = harmful_df.sample(n=min(n_samples, len(harmful_df)), random_state=42)

    # Combine and shuffle
    batch = pd.concat([benign_batch, harmful_batch], ignore_index=True)
    batch = batch.sample(frac=1, random_state=42).reset_index(drop=True)

    return batch
