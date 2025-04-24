# visQAI/src/db/__init__.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

engine = create_engine('sqlite:///visq_database.db', echo=False)
Session = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)
