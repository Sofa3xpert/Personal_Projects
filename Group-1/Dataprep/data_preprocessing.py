import pandas as pd
import re
import logging
import sys

# Global Constants
REQUIRED_COLUMNS = ["statement", "status"]
ALLOWED_LABELS = [
    "Normal", "Depression", "Suicidal", "Anxiety",
    "Stress", "Bi-Polar", "Personality Disorder"
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    Parameters:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        raise


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that all required columns are present in the DataFrame.
    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
    Returns:
        pd.DataFrame: The same DataFrame if validation passes.
    Raises:
        ValueError: If any required column is missing.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    logging.info("All required columns are present.")
    return df


def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and enforce that 'statement' and 'status' columns are of type object (string).
    Parameters:
        df (pd.DataFrame): The DataFrame to check.
    Returns:
        pd.DataFrame: The DataFrame with corrected data types if necessary.
    """
    if not pd.api.types.is_object_dtype(df["statement"]):
        logging.warning("Column 'statement' is not of type object. Converting to string.")
        df["statement"] = df["statement"].astype(str)

    if not pd.api.types.is_object_dtype(df["status"]):
        logging.warning("Column 'status' is not of type object. Converting to string.")
        df["status"] = df["status"].astype(str)

    logging.info("Data types for 'statement' and 'status' have been validated.")
    return df


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where 'statement' or 'status' are missing.
    Parameters:
        df (pd.DataFrame): The DataFrame to process.
    Returns:
        pd.DataFrame: The DataFrame with missing rows removed.
    """
    initial_shape = df.shape
    df = df.dropna(subset=["statement", "status"])
    dropped = initial_shape[0] - df.shape[0]
    logging.info(f"Dropped {dropped} rows with missing 'statement' or 'status'.")
    return df


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the 'status' column contains only allowed labels.
    Rows with invalid labels are dropped.
    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
    Returns:
        pd.DataFrame: The DataFrame containing only rows with valid labels.
    """
    invalid_rows = df[~df["status"].isin(ALLOWED_LABELS)]
    if not invalid_rows.empty:
        logging.warning(f"Found {invalid_rows.shape[0]} rows with invalid mental health labels. Dropping these rows.")
        df = df[df["status"].isin(ALLOWED_LABELS)]
    else:
        logging.info("All mental health status labels are valid.")
    return df


def clean_text(text: str) -> str:
    """
    Perform basic text cleaning on a given string.
    Cleaning steps:
      - Convert to lowercase.
      - Remove URLs.
      - Remove unwanted characters (only alphanumerics, whitespace, and basic punctuation are kept).
      - Reduce repeated punctuation to a single instance.
      - Remove extra whitespace.
    Parameters:
        text (str): The original text.
    Returns:
        str: The cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs (e.g., http://... or www...)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove unwanted characters:
    text = re.sub(r'[^a-z0-9\s\.,\?!\']', '', text)

    # Reduce repeated punctuation (e.g., "!!!" -> "!")
    text = re.sub(r'([.,?!]){2,}', r'\1', text)

    # Replace multiple whitespace characters with a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the clean_text function to the 'statement' column of the DataFrame.
    Logs a random sample of statements before and after cleaning for validation.
    Parameters:
        df (pd.DataFrame): The DataFrame with a 'statement' column.
    Returns:
        pd.DataFrame: The DataFrame with the cleaned 'statement' column.
    """
    # Log a sample of original statements
    sample_size = min(5, len(df))
    logging.info("Sample statements before cleaning:")
    logging.info(df['statement'].sample(sample_size).tolist())

    # Clean the text in the 'statement' column
    df['statement'] = df['statement'].apply(clean_text)

    # Log a sample of cleaned statements
    logging.info("Sample statements after cleaning:")
    logging.info(df['statement'].sample(sample_size).tolist())

    return df


def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and drop duplicate rows based on the 'statement' column.
    The first occurrence is retained.
    Parameters:
        df (pd.DataFrame): The DataFrame to deduplicate.
    Returns:
        pd.DataFrame: The deduplicated DataFrame.
    """
    initial_shape = df.shape
    df = df.drop_duplicates(subset=["statement"], keep="first")
    duplicates_dropped = initial_shape[0] - df.shape[0]
    logging.info(f"Dropped {duplicates_dropped} duplicate rows based on 'statement'.")
    return df


def save_clean_data(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the cleaned DataFrame to a CSV file.
    Parameters:
        df (pd.DataFrame): The cleaned DataFrame.
        output_file (str): The output file path.
    """
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Clean data saved successfully to {output_file}")
    except Exception as e:
        logging.error(f"Error saving clean data to {output_file}: {e}")
        raise


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    # Define input and output file paths (update input_file as needed)
    input_file = r"/Group-1/Dataprep/Combined Data.csv"
    output_file = "cleaned_data.csv"

    try:
        df = load_dataset(input_file)
        df = validate_columns(df)
        df = check_data_types(df)
        df = drop_missing_rows(df)
        df = validate_labels(df)
        df = clean_text_column(df)
        df = deduplicate_data(df)
        logging.info(f"Final DataFrame shape after preprocessing: {df.shape}")
        save_clean_data(df, output_file)
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
