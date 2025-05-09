from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class IsolationResistance(Base):
    __tablename__ = 'isolation_resistance'

    id = Column(Integer, primary_key=True)
    resistance = Column(Float, nullable=False)
    voltage = Column(Float, nullable=False)
    test_time = Column(DateTime, nullable=False)
    result = Column(String(50), nullable=False)
    description = Column(String(255))

    def __repr__(self):
        return f"<IsolationResistance(id={self.id}, resistance={self.resistance}, voltage={self.voltage}, result='{self.result}')>"
