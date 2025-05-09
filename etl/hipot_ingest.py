from etl.common import load_data, normalize_data
from models.hipot import HiPotData
from sqlalchemy.orm import Session
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_hipot_data(file_path: str, session: Session) -> None:
    """
    Ingest HiPot test data from a CSV file into the database.
    
    Args:
        file_path (str): Path to the CSV file containing HiPot data
        session (Session): SQLAlchemy database session
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        # Normalize the data
        normalized_df = normalize_data(df, test_type='hipot')
        
        # Ensure data types are correct
        normalized_df['id'] = normalized_df['id'].astype(int)
        normalized_df['voltage'] = normalized_df['voltage'].astype(float)
        normalized_df['current'] = normalized_df['current'].astype(float)
        normalized_df['test_time'] = pd.to_datetime(normalized_df['test_time'])
        normalized_df['result'] = normalized_df['result'].astype(str)
        
        # Convert DataFrame to list of HiPotData objects
        hipot_data_objects = [
            HiPotData(
                id=row['id'],
                voltage=row['voltage'],
                current=row['current'],
                test_time=row['test_time'],
                result=row['result'],
                description=f"HiPot Test {row['id']}"
            ) for index, row in normalized_df.iterrows()
        ]
        
        # Bulk insert into the database
        session.bulk_save_objects(hipot_data_objects)
        session.commit()
        
        logger.info(f"Successfully ingested {len(hipot_data_objects)} HiPot data records.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting HiPot data: {str(e)}")
        raise
