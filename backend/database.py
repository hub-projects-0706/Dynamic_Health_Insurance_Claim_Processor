import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

raw_db_url = os.getenv("DB_URL", "postgresql://claims_user:claims_password@postgres:5432/claims_db")

# Strip JDBC prefix if present
if raw_db_url.startswith("jdbc:postgresql://"):
    raw_db_url = raw_db_url.replace("jdbc:postgresql://", "postgresql://")
elif raw_db_url.startswith("jdbc:h2:"):
    raw_db_url = "sqlite:///./claims.db"

# Fallback construction from individual env vars if default URL is not valid
db_user = os.getenv("DB_USER", "claims_user")
db_pass = os.getenv("DB_PASS", "claims_password")
db_host = os.getenv("DB_HOST", "postgres")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "claims_db")

if not raw_db_url:
    raw_db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

engine_kwargs = {}
if raw_db_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

try:
    engine = create_engine(raw_db_url, **engine_kwargs)
except Exception as e:
    # Graceful fallback to local SQLite database if PostgreSQL fails to initialize initially
    print(f"Warning: Failed to create engine with URL {raw_db_url}: {e}. Falling back to SQLite.")
    raw_db_url = "sqlite:///./claims.db"
    engine = create_engine(raw_db_url, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
