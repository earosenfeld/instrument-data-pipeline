import pytest
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importing the package registers EVERY model on the single shared Base, so
# create_all() below builds all six tables (not just one).
from models import (
    Base,
    BurnInZeroCurrent,
    ICTData,
    ParametricData,
    LaserProfile,
    HiPotData,
    IsolationResistance,
)


# Fixture for setting up an in-memory SQLite database. Function scope gives each
# test a clean database so ingest tests do not see each other's rows.
@pytest.fixture(scope='function')
def in_memory_db():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)  # Create ALL tables
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

# Fixture for loading sample ICT data
@pytest.fixture(scope='module')
def sample_ict_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'voltage': [5.0, 5.1, 5.2],
        'current': [0.1, 0.11, 0.12],
        'test_time': [datetime.now()] * 3,
        'result': ['PASS', 'PASS', 'PASS'],
        'description': ['ICT Test 1', 'ICT Test 2', 'ICT Test 3']
    })
    return data

# Fixture for loading sample burn-in data
@pytest.fixture(scope='module')
def sample_burnin_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [50.0, 60.0, 70.0],
        'description': ['Burn-in Test 1', 'Burn-in Test 2', 'Burn-in Test 3']
    })
    return data

# Fixture for loading sample HiPot data
@pytest.fixture(scope='module')
def sample_hipot_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'voltage': [1000.0, 1100.0, 1200.0],
        'current': [0.001, 0.0012, 0.0015],
        'test_time': [datetime.now()] * 3,
        'result': ['PASS', 'PASS', 'PASS'],
        'description': ['HiPot Test 1', 'HiPot Test 2', 'HiPot Test 3']
    })
    return data

# Fixture for loading sample isolation resistance data
@pytest.fixture(scope='module')
def sample_isolation_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'resistance': [1000000.0, 1100000.0, 1200000.0],
        'voltage': [500.0, 550.0, 600.0],
        'test_time': [datetime.now()] * 3,
        'result': ['PASS', 'PASS', 'PASS'],
        'description': ['Isolation Test 1', 'Isolation Test 2', 'Isolation Test 3']
    })
    return data

# Fixture for loading sample laser profile data
@pytest.fixture(scope='module')
def sample_laser_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'power': [10.0, 11.0, 12.0],
        'wavelength': [1550.0, 1551.0, 1552.0],
        'test_time': [datetime.now()] * 3,
        'result': ['PASS', 'PASS', 'PASS'],
        'description': ['Laser Test 1', 'Laser Test 2', 'Laser Test 3']
    })
    return data

# Fixture for loading sample parametric data
@pytest.fixture(scope='module')
def sample_parametric_data():
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'voltage': [3.3, 3.4, 3.5],
        'current': [0.5, 0.55, 0.6],
        'test_time': [datetime.now()] * 3,
        'result': ['PASS', 'PASS', 'PASS'],
        'description': ['Parametric Test 1', 'Parametric Test 2', 'Parametric Test 3']
    })
    return data

# Additional fixtures for other data types can be added similarly
