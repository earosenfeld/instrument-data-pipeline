from etl.common import load_data, normalize_data
from models.isolation import IsolationResistance
from sqlalchemy.orm import Session
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_isolation_resistance_data(file_path: str, session: Session) -> None:
    """
    Ingest isolation resistance test data from a CSV file into the database.
    
    Args:
        file_path (str): Path to the CSV file containing isolation resistance data
        session (Session): SQLAlchemy database session
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        # Normalize the data
        normalized_df = normalize_data(df, test_type='isolation')
        
        # Ensure data types are correct
        normalized_df['id'] = normalized_df['id'].astype(int)
        normalized_df['resistance'] = normalized_df['resistance'].astype(float)
        normalized_df['voltage'] = normalized_df['voltage'].astype(float)
        normalized_df['test_time'] = pd.to_datetime(normalized_df['test_time'])
        normalized_df['result'] = normalized_df['result'].astype(str)
        
        # Convert DataFrame to list of IsolationResistance objects
        isolation_resistance_objects = [
            IsolationResistance(
                id=row['id'],
                resistance=row['resistance'],
                voltage=row['voltage'],
                test_time=row['test_time'],
                result=row['result'],
                description=f"Isolation Test {row['id']}"
            ) for index, row in normalized_df.iterrows()
        ]
        
        # Bulk insert into the database
        session.bulk_save_objects(isolation_resistance_objects)
        session.commit()
        
        logger.info(f"Successfully ingested {len(isolation_resistance_objects)} isolation resistance data records.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting isolation resistance data: {str(e)}")
        raise
