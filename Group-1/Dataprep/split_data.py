import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import sys


def load_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from a CSV file.
    Parameters:
        file_path (str): Path to the cleaned CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Cleaned dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading cleaned dataset from {file_path}: {e}")
        raise


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.70,
                  val_ratio: float = 0.15, test_ratio: float = 0.15,
                  random_state: int = 42):
    """
    Split the DataFrame into training, validation, and testing sets with stratification
    on the 'status' column.
    Parameters:
        df (pd.DataFrame): The cleaned DataFrame to split.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducibility.
    Returns:
        tuple: (train_df, val_df, test_df)
    Raises:
        ValueError: If the provided ratios are not between 0 and 1 or do not sum to 1.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    # First split: separate training data from the rest.
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["status"],
        shuffle=True,
        random_state=random_state
    )

    # Second split: split the remaining data into validation and testing sets.
    # The ratio of validation within the temp_df is: val_ratio / (val_ratio + test_ratio)
    val_proportion = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_proportion),
        stratify=temp_df["status"],
        shuffle=True,
        random_state=random_state
    )

    return train_df, val_df, test_df


def save_split_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    train_file: str = "train_data.csv",
                    val_file: str = "val_data.csv",
                    test_file: str = "test_data.csv") -> None:
    """
    Save the train, validation, and test DataFrames to CSV files.
    Parameters:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Testing set.
        train_file (str): Output file path for training data.
        val_file (str): Output file path for validation data.
        test_file (str): Output file path for testing data.
    """
    try:
        train_df.to_csv(train_file, index=False)
        logging.info(f"Training data saved successfully to {train_file}. Shape: {train_df.shape}")
        val_df.to_csv(val_file, index=False)
        logging.info(f"Validation data saved successfully to {val_file}. Shape: {val_df.shape}")
        test_df.to_csv(test_file, index=False)
        logging.info(f"Testing data saved successfully to {test_file}. Shape: {test_df.shape}")
    except Exception as e:
        logging.error(f"Error saving split data: {e}")
        raise


def print_split_info(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Print shapes and label distributions for each split.
    Parameters:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Testing set.
    """
    logging.info(f"Train set shape: {train_df.shape}")
    logging.info(f"Validation set shape: {val_df.shape}")
    logging.info(f"Test set shape: {test_df.shape}")

    logging.info("Train set label distribution:\n" + str(train_df["status"].value_counts()))
    logging.info("Train set label percentages:\n" + str(train_df["status"].value_counts(normalize=True)))

    logging.info("Validation set label distribution:\n" + str(val_df["status"].value_counts()))
    logging.info("Validation set label percentages:\n" + str(val_df["status"].value_counts(normalize=True)))

    logging.info("Test set label distribution:\n" + str(test_df["status"].value_counts()))
    logging.info("Test set label percentages:\n" + str(test_df["status"].value_counts(normalize=True)))


def main():
    """
    Main function to run the data splitting pipeline.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Define input file path and output file names
    input_file = r"/Group-1/Dataprep/cleaned_data.csv"

    try:
        df = load_clean_data(input_file)
        train_df, val_df, test_df = split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                                                  random_state=42)
        print_split_info(train_df, val_df, test_df)
        save_split_data(train_df, val_df, test_df)

    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()