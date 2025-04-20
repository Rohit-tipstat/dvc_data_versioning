import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt")
import logging

# ensure the "logs" directory exists
log_dirs = "logs"
os.makedirs(log_dirs, exist_ok=True)

# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dirs, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    """
    Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctuations, and stemming.
    """
    ps = PorterStemmer()
    # convert text to lowercase
    text = text.lower()
    # tokenize the text
    text = nltk.word_tokenize(text)
    # remove non-alphabetic tokens
    text = [word for word in text if word.isalnum()]
    # remove stopword and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # stem the words
    text = [ps.stem(word) for word in text]
    # join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df: pd.DataFrame, text_column='text', target_column='target') -> pd.DataFrame:
    """
    Preprocess the Dataframe by encoding the target column, removing duplicates, and transforming the text column
    """
    try:
        logger.debug("Starting preprocessing for Dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")
        
        # removing duplicate rows
        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")
        
        # apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        
        return df
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    
def main(text_column="text", target_column="target"):
    """
    Main function that loads raw data, preprocess it and save the processed data
    """
    try:
        # fetching data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")
        
        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)
        
        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False) 
        logger.debug("Processed data saved to %s", data_path)   
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()