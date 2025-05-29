import pandas as pd
import logging

# Configure logging to track script execution and errors
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

#Define required columns for the dataset validation
required_columns = ["Patient ID",  # Unique identifier for each patient
    "Age",  # Age of the patient
    "Gender",  # Male or Female
    "Diagnosis",  # Mental health condition (e.g., Anxiety, Depression)
    "Symptom Severity (1-10)",  # Severity of symptoms
    "Mood Score (1-10)",  # Mood rating during treatment
    "Sleep Quality (1-10)",  # Patient-reported sleep quality
    "Physical Activity (hrs/week)",  # Hours per week of activity
    "Medication",  # Medications prescribed (e.g., SSRIs, Antidepressants)
    "Therapy Type",  # Type of therapy (e.g., CBT, DBT)
    "Treatment Start Date",  # Date treatment started
    "Treatment Duration (weeks)",  # Duration of treatment in weeks
    "Stress Level (1-10)",  # Patient's stress level
    "Outcome",  # Treatment outcome (e.g., Improved, Deteriorated)
    "Treatment Progress (1-10)",  # Progress made during treatment
    "AI-Detected Emotional State",  # AI-detected emotional state (e.g., Happy, Anxious)
    "Adherence to Treatment (%)"  # Percentage of adherence to treatment plan
]

#Define valid numerical ranges for specific columns
numeric_columns = {
    'Age': (0, 120),                          # Age can range from 0 to 120
    'Symptom Severity (1-10)': (1, 10),       # Symptom Severity should be between 1 and 10
    'Mood Score (1-10)': (1, 10),             # Mood Score should be between 1 and 10
    'Sleep Quality (1-10)': (1, 10),          # Sleep Quality should be between 1 and 10
    'Physical Activity (hrs/week)': (0, 168), # Physical Activity can range from 0 to 168 hours/week
    'Treatment Duration (weeks)': (0, 104),   # Treatment Duration can range from 0 to 104 weeks (2 years)
    'Stress Level (1-10)': (1, 10),           # Stress Level should be between 1 and 10
    'Treatment Progress (1-10)': (1, 10),     # Treatment Progress should be between 1 and 10
    'Adherence to Treatment (%)': (0, 100)    # Adherence to Treatment should be between 0% and 100%
}

#Define allowed categorical values for specific columns
categorical_columns = {
    'Gender': ['Male', 'Female'],                       # Gender can only be "Male" or "Female"
    'Diagnosis': ['Generalized Anxiety', 'Major Depressive Disorder', 'Bipolar Disorder', 'Panic Disorder'],  # Allowed diagnoses
    'Medication': ['Benzodiazepines', 'SSRIs', 'Mood Stabilizer', 'Antipsychotics', 'Antidepressants' ],         # Allowed medications
    'Therapy Type': ['Interpersonal Therapy', 'Mindfulness-Based Therapy','Cognitive Behavioral Therapy','Dialectical Behavioral Therapy'], # Allowed therapy types
    'AI-Detected Emotional State': ['Happy', 'Stressed', 'Anxious', 'Excited', 'Neutral'],  # Allowed emotional states
    'Outcome': ['Deteriorated','No Change', 'Improved'] #Allowed outcome
}

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        df (pd.DataFrame): Loaded DataFrame
        None: If an error occurs
    """
    
    try:
        logger.info(f"Start loading dataset {file_path}") 

        df = pd.read_csv(file_path) 

        logger.info(f"Successfully loaded the dataset: {file_path}") 

        return df 
    
    except Exception as e:

        logger.error(f"Error loading dataset {file_path}: {str(e)}") 

        return None 

def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        df (pd.DataFrame): DataFrame with duplicates removed.
        None: If an error occurs
    """
    try:
        duplicate_rows_before = df.duplicated().sum() 

        df = df.drop_duplicates() 

        duplicate_rows_after = df.duplicated().sum() 

        duplicates_removed = duplicate_rows_before - duplicate_rows_after 

        logger.info(f"Removed {duplicates_removed} duplicate rows.") 

        return df 

    except Exception as e:
         
         logger.error(f"Error removing duplicates: {str(e)}")

         return None

def exclude_incomplete_rows(df: pd.DataFrame,required_columns: list) -> pd.DataFrame:
    """
    Remove rows with missing values in required columns.

    Parameters:
        df (pd.DataFrame) : Input DataFrame.
        required_columns (list): List of required columns.

    Returns:
        df (pd.DataFrame)Cleaned DataFrame.
        None: If an error occurs
    """
    try: 
        initial_rows = len(df) 
        
        df = df.dropna(subset=required_columns) 
        
        excluded_rows = initial_rows - len(df) 
        
        logger.info(f"Dropped {excluded_rows} rows with missing required columns") 

        return df
    
    except Exception as e:
        
        logger.error(f"Error excluding incomplete rows: {str(e)}")

        return None



def remove_invalid_range(df: pd.DataFrame, numeric_columns: dict) -> pd.DataFrame:
    """
    Remove rows with numerical values outside of allowed ranges.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        numeric_columns (dict): Dictionary of valid numerical ranges.

    Returns:
        df (pd.DataFrame): Cleaned DataFrame.
        None: If an error occurs.
    """
    try:
        initial_rows = len(df)

        for col, (min_value, max_value) in numeric_columns.items():

        
            if col in df.columns:

                invalid = (df[col] < min_value) | (df[col] > max_value)

                invalid_rows = df[invalid]

            if not invalid_rows.empty:
                
                logger.warning(f"Found invalid values in column '{col}'")

                logger.warning(invalid_rows)
        
        removed_rows = initial_rows - len(df)

        logger.info(f"{removed_rows} rows with invalid range values have been removed")

        return df

    except Exception as e:
        logger.error(f"Error removing invalid range values: {str(e)}")

        return None

def remove_invalid_categorical_values(df: pd.DataFrame, categorical_columns: dict) -> pd.DataFrame:
    """
    Remove rows with invalid categorical values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        categorical_columns (dict): Dictionary of allowed categorical values.
    
    Returns:
        df (pd.DataFrame): Cleaned DataFrame.
        None: If an error occurs.
    """

    try:

        initial_rows = len(df)

        for col, values in categorical_columns.items():
            if col in df.columns:

                invalid = ~df[col].isin(values)

                invalid_rows = df[invalid]

                if not invalid_rows.empty:
                    logger.warning(f"Found rows with invalid values in column '{col}' that are not in {categorical_columns}:")
                    logger.warning(invalid_rows)
                
                df = df[~invalid]
            
            removed_rows = initial_rows - len(df)
            
            logger.info(f"{removed_rows} Rows with invalid categorical values have been removed.")

            return df
    
    except Exception as e:
        logger.error(f"Error removing invalid categorical values: {str(e)}")

        return None

def map_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map diagnoses to match those in the first (prediction) dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        df (pd.DataFrame): Updated DataFrame with mapped diagnosis values.
    """
    try:
        def mapping(row):
            if row['Diagnosis'] == "Generalized Anxiety":
                return "Anxiety"
            elif row["Diagnosis"] == "Major Depressive Disorder":
                return "Suicidal" if row["Symptom Severity (1-10)"] >= 8 else "Depression"
            elif row["Diagnosis"] == "Bipolar Disorder":
                return "Bipolar"
            elif row["Diagnosis"] == "Panic Disorder":
                return "Anxiety"
            else:
                return row["Diagnosis"]
            
        df["Diagnosis"] = df.apply(mapping,axis=1)

        logger.info(f"Diagnosis column successfully mapped. {df['Diagnosis'].unique()}")

        return df

    except Exception as e:
        logger.error(f"Error mapping diagnosis: {str(e)}")

        return None

def save_dataset(df: pd.DataFrame, file_path: str):
    """
    Save the processed dataset to a CSV file.

    Parameters:
        df (pd.DataFrame): Processed DataFrame to save.
        file_path (str): Path to the CSV file.

    Returns:
        None. Logs success or failure of save operation.
    """

    try:
        df.to_csv(file_path,index=False)

        logger.info(f"Dataset successfully saved to {file_path}")

    except:
        logger.error(f"Dataset successfully saved to {file_path}")

def main():

    """
    Main function to execute the data preprocessing workflow.
    
    Returns:
        None. Logs errors if any step in the preprocessing pipeline fails.
    """

    try:
        df = load_data("mental_health_diagnosis_treatment_.csv")
        if df is None:
            return
        df = check_duplicates(df)
        df = exclude_incomplete_rows(df,required_columns)
        df = remove_invalid_range(df, numeric_columns)
        df = remove_invalid_categorical_values(df, categorical_columns)
        df = map_diagnosis(df)
        save_dataset(df, "cleaned_dataset.csv")

        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
    

if __name__ == "__main__":
    main()
    