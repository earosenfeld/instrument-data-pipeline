from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LaserProfile(Base):
    __tablename__ = 'laser_profile'

    id = Column(Integer, primary_key=True)
    power = Column(Float, nullable=False)
    wavelength = Column(Float, nullable=False)
    test_time = Column(DateTime, nullable=False)
    result = Column(String(50), nullable=False)
    description = Column(String(255))

    def __repr__(self):
        return f"<LaserProfile(id={self.id}, power={self.power}, wavelength={self.wavelength}, result='{self.result}')>"
