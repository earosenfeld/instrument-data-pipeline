from etl.common import load_data, normalize_data
from models.ict import ICTData
from sqlalchemy.orm import Session
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_ict_data(file_path: str, session: Session) -> None:
    """Ingest ICT data from a CSV file into the database.

    Args:
        file_path: Path to the CSV file containing ICT data.
        session: SQLAlchemy database session.
    """
    try:
        df = load_data(file_path)
        normalized_df = normalize_data(df, test_type='ict')

        normalized_df['id'] = normalized_df['id'].astype(int)
        normalized_df['voltage'] = normalized_df['voltage'].astype(float)
        normalized_df['current'] = normalized_df['current'].astype(float)
        normalized_df['test_time'] = pd.to_datetime(normalized_df['test_time'])
        normalized_df['result'] = normalized_df['result'].astype(str)

        ict_data_objects = [
            ICTData(
                id=row['id'],
                voltage=row['voltage'],
                current=row['current'],
                test_time=row['test_time'],
                result=row['result'],
                description=f"ICT Test {row['id']}",
            )
            for _, row in normalized_df.iterrows()
        ]

        session.bulk_save_objects(ict_data_objects)
        session.commit()
        logger.info(f"Successfully ingested {len(ict_data_objects)} ICT data records.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error ingesting ICT data: {str(e)}")
        raise
