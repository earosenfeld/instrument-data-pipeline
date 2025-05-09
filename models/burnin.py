from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class BurnInZeroCurrent(Base):
    __tablename__ = 'burnin_zero_current'

    id = Column(Integer, primary_key=True)
    value = Column(Float, nullable=False)
    description = Column(String(255), nullable=False)

    def __repr__(self):
        return f"<BurnInZeroCurrent(id={self.id}, value={self.value}, description='{self.description}')>"
