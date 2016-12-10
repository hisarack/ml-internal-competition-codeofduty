from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float
import sys
import numpy as np


def calculate(probs):
        tmp = np.array(probs).astype('float32')
        tmp = np.where(tmp > 0.00001)
        return np.std(tmp)

Base = declarative_base()


class Digit(Base):
    __tablename__ = 'digit'

    fname = Column(String(32), primary_key=True)
    is0 = Column(Float)
    is1 = Column(Float)
    is2 = Column(Float)
    is3 = Column(Float)
    is4 = Column(Float)
    is5 = Column(Float)
    is6 = Column(Float)
    is7 = Column(Float)
    is8 = Column(Float)
    is9 = Column(Float)
    std = Column(Float)

engine = create_engine("mysql+mysqldb://root:billychu21@localhost/codeofduty")

Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

session.query(Digit).delete()

with open(sys.argv[1]) as f:
    for line in f:
        tmp = line.strip().split(",")
        maxval = max(np.array(tmp[1:]).astype('float32'))
        if maxval > 0.875:
            continue

        std = calculate(tmp[1:])
        digit_obj = Digit(fname=tmp[0])
        digit_obj.is0 = float(tmp[1])
        digit_obj.is1 = float(tmp[2])
        digit_obj.is2 = float(tmp[3])
        digit_obj.is3 = float(tmp[4])
        digit_obj.is4 = float(tmp[5])
        digit_obj.is5 = float(tmp[6])
        digit_obj.is6 = float(tmp[7])
        digit_obj.is7 = float(tmp[8])
        digit_obj.is8 = float(tmp[9])
        digit_obj.is9 = float(tmp[10])
        digit_obj.std = std
        session.add(digit_obj)
        session.flush()
    session.commit()
session.close()
f.close()
