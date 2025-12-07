from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./souschef.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # required for SQLite + Streamlit
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String)        # pantry / fridge / freezer
    quantity = Column(Float)
    unit = Column(String)
    purchase_date = Column(Date)
    best_buy_date = Column(Date, nullable=True)
    best_buy_source = Column(String, default="user")  # user / ai
    last_updated = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
