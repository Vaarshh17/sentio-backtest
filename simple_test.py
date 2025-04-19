# Create a simple test script
with open('C:/Users/user/sentio-backtest/test_import.py', 'w') as f:
    f.write("""
# Simple import test
print("Attempting to import from dummy_modules...")
from dummy_modules import DummyDataManager, DummyDataSource

print("Success!")
dm = DummyDataManager()
ds = DummyDataSource()
""")