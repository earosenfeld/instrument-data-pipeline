from etl.common import load_data, normalize_data
from models.parametric import ParametricData
from sqlalchemy.orm import Session
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_parametric_data(file_path: str, session: Session) -> None:
    """
    Ingest parametric test data from a CSV file into the database.
    
    Args:
        file_path (str): Path to the CSV file containing parametric data
        session (Session): SQLAlchemy database session
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        # Normalize the data
        normalized_df = normalize_data(df, test_type='parametric')
        
        # Ensure data types are correct
        normalized_df['id'] = normalized_df['id'].astype(int)
        normalized_df['voltage'] = normalized_df['voltage'].astype(float)
        normalized_df['current'] = normalized_df['current'].astype(float)
        normalized_df['test_time'] = pd.to_datetime(normalized_df['test_time'])
        normalized_df['result'] = normalized_df['result'].astype(str)
        
        # Convert DataFrame to list of ParametricData objects
        parametric_data_objects = [
            ParametricData(
                id=row['id'],
                voltage=row['voltage'],
                current=row['current'],
                test_time=row['test_time'],
                result=row['result'],
                description=f"Parametric Test {row['id']}"
            ) for index, row in normalized_df.iterrows()
        ]
        
        # Bulk insert into the database
        session.bulk_save_objects(parametric_data_objects)
        session.commit()
        
        logger.info(f"Successfully ingested {len(parametric_data_objects)} parametric data records.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting parametric data: {str(e)}")
        raise
