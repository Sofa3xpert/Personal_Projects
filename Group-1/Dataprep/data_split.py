import pandas as pd
import logging
from sklearn.model_selection import train_test_split

#Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define format
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("app.log")  # Output to a file
    ]
)

#Create logger instance
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        df (pd.DataFrame): Loaded dataset as a Pandas DataFrame, or None if an error occurs.
    """
    
    try:
        logger.info(f"Start loading dataset {file_path}")

        df = pd.read_csv(file_path)

        logger.info(f"Successfully loaded the dataset: {file_path}")

        return df
    
    except Exception as e:

        logger.error(f"Error loading dataset {file_path}: {str(e)}")

        return None

def create_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the dataset into training, validation and testing datasets.
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        train_df (pd.DataFrame): 

    """
    try:

        train_df, temp_df = train_test_split(df,test_size= 0.3,random_state =42)

        val_df,test_df = train_test_split(temp_df,test_size=0.5,random_state=42)

        logger.info(f"Training Dataset Shape {train_df.shape}")
        logger.info(f"Testing Dataset Shape {test_df.shape}")
        logger.info(f"Validation Dataset Shape {val_df.shape}")

        return train_df,test_df,val_df
    
    except Exception as e:

        logger.error(f"Error splitting dataset: {str(e)}")

        return None

def save_datasets(train_df: pd.DataFrame,test_df: pd.DataFrame,val_df: pd.DataFrame) -> None:
    """
    Save the processed dataset to a CSV file.

    Parameters:
        file_path (str): Path to CSV file.

    Returns:
        None. Logs success or failure to save operation.

    """
    try:

        train_dataset = train_df.to_csv('train_dataset.csv', index=False)

        logger.info(f"Saved Training Dataset")

        test_dataset = test_df.to_csv('test_dataset.csv', index=False)

        logger.info(f"Saved Testing Data")

        val_dataset = val_df.to_csv('val_dataset.csv', index=False)

        logger.info(f"Saved Validation Dataset")
    
    except Exception as e:

        logger.error(f"Error saving datasets: {str(e)}")

        return None

def main():

    """
    Main function to execute the data loading, splitting and saving workflow.
    """

    try:
        df = load_data("cleaned_dataset.csv")
        train_df,test_df,val_df = create_datasets(df)
        save_datasets(train_df,test_df,val_df)

    except Exception as e:

        logger.error(f"Error splitting and saving datasets: {str(e)} ")



if __name__ == "__main__":
    main()
    

