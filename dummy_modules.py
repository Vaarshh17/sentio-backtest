# Create a simple version of dummy_modules.py
with open('C:/Users/user/sentio-backtest/dummy_modules.py', 'w') as f:
    f.write("""
# Basic dummy modules
class DummyDataManager:
    def __init__(self):
        print("DummyDataManager initialized")

class DummyDataSource:
    def __init__(self):
        print("DummyDataSource initialized")
""")