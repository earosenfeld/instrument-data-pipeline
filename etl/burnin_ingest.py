from etl.common import load_data, normalize_data
from models.burnin import BurnInZeroCurrent
from sqlalchemy.orm import Session
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_burnin_zero_current_data(file_path: str, session: Session) -> None:
    """
    Ingest burn-in zero current data from a CSV file into the database.
    
    Args:
        file_path (str): Path to the CSV file containing burn-in data
        session (Session): SQLAlchemy database session
    """
    try:
        # Load the data
        df = load_data(file_path)

        # Normalize the data
        normalized_df = normalize_data(df)

        # Empty input is valid: nothing to ingest, no error.
        if normalized_df.empty:
            logger.info("No rows to ingest from %s (empty file).", file_path)
            return

        # Ensure data types are correct. A malformed file (non-numeric value or
        # ragged rows producing NaN) raises here, which callers expect.
        normalized_df['value'] = normalized_df['value'].astype(float)
        normalized_df['id'] = normalized_df['id'].astype(int)

        # Convert DataFrame to list of BurnInZeroCurrent objects
        burnin_zero_current_objects = [
            BurnInZeroCurrent(
                id=row['id'],
                value=row['value'],
                description=f"Burn-in Test {row['id']}"
            ) for index, row in normalized_df.iterrows()
        ]
        
        # Bulk insert into the database
        session.bulk_save_objects(burnin_zero_current_objects)
        session.commit()
        
        logger.info(f"Successfully ingested {len(burnin_zero_current_objects)} burn-in zero current data records.")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting burn-in zero current data: {str(e)}")
        raise
