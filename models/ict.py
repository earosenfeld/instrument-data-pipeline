from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ICTData(Base):
    __tablename__ = 'ict_data'

    id = Column(Integer, primary_key=True)
    voltage = Column(Float, nullable=False)
    current = Column(Float, nullable=False)
    test_time = Column(DateTime, nullable=False)
    result = Column(String(50), nullable=False)
    description = Column(String(255))

    def __repr__(self):
        return f"<ICTData(id={self.id}, voltage={self.voltage}, current={self.current}, result='{self.result}')>"
