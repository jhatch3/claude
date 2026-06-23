"""`python -m src.db` — create the tables and seed the mock data."""
from src.db.database import db
from src.db.seed import seed

seed()
print(f"Database ready at {db.url}")
