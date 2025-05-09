from etl.common import load_data, normalize_data
from models.laser import LaserProfile
from sqlalchemy.orm import Session
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_laser_profile_data(file_path: str, session: Session) -> None:
    """
    Ingest laser profile test data from a CSV file into the database.
    
    Args:
        file_path (str): Path to the CSV file containing laser profile data
        session (Session): SQLAlchemy database session
    """
    try:
        # Load the data
        df = load_data(file_path)
        
        # Normalize the data
        normalized_df = normalize_data(df, test_type='laser')
        
        # Ensure data types are correct
        normalized_df['id'] = normalized_df['id'].astype(int)
        normalized_df['power'] = normalized_df['power'].astype(float)
        normalized_df['wavelength'] = normalized_df['wavelength'].astype(float)
        normalized_df['test_time'] = pd.to_datetime(normalized_df['test_time'])
        normalized_df['result'] = normalized_df['result'].astype(str)
        
        # Convert DataFrame to list of LaserProfile objects
        laser_profile_objects = [
            LaserProfile(
                id=row['id'],
                power=row['power'],
                wavelength=row['wavelength'],
                test_time=row['test_time'],
                result=row['result'],
                description=f"Laser Test {row['id']}"
            ) for index, row in normalized_df.iterrows()
        ]
        
        # Bulk insert into the database
        session.bulk_save_objects(laser_profile_objects)
        session.commit()
        
        logger.info(f"Successfully ingested {len(laser_profile_objects)} laser profile data records.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting laser profile data: {str(e)}")
        raise
