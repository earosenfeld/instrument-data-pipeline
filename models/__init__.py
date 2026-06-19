"""ORM models for the instrument data pipeline.

All model modules share the SINGLE declarative ``Base`` defined here. Previously
each module called ``declarative_base()`` independently, so each class registered
on a different metadata object and ``Base.metadata.create_all()`` built only one
table. Importing the one ``Base`` from this package fixes that: every model is
registered on the same metadata and ``create_all`` builds them all.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()

# Import every model so that merely importing ``models`` (or this package's
# ``Base``) populates ``Base.metadata`` with all tables.
from models.burnin import BurnInZeroCurrent  # noqa: E402,F401
from models.hipot import HiPotData  # noqa: E402,F401
from models.ict import ICTData  # noqa: E402,F401
from models.isolation import IsolationResistance  # noqa: E402,F401
from models.laser import LaserProfile  # noqa: E402,F401
from models.parametric import ParametricData  # noqa: E402,F401

__all__ = [
    "Base",
    "BurnInZeroCurrent",
    "HiPotData",
    "ICTData",
    "IsolationResistance",
    "LaserProfile",
    "ParametricData",
]
