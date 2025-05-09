from etl.common import load_data, normalize_data
from models.ict import ICTData
from sqlalchemy.orm import Session

def ingest_ict_data(file_path, session: Session):
    # Load the data
    df = load_data(file_path)
    
    # Normalize the data
    normalized_df = normalize_data(df)
    
    # Convert DataFrame to list of ICTData objects
    ict_data_objects = [ICTData(id=row['id'], value=row['value'], description='Sample ICT data') for index, row in normalized_df.iterrows()]
    
    # Bulk insert into the database
    session.bulk_save_objects(ict_data_objects)
    session.commit()
    
    print(f"Ingested {len(ict_data_objects)} ICT data records.")
