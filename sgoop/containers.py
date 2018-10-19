import os.path as op
import pandas as pd


def read_plumed_file(filename):
    """
    Read PLUMED output files into pandas DataFrame, with colvars/bias column names
    and time indices.

    :param filename:
    :return: pd.DataFrame
    """
    filename = op.abspath(filename)

    with open(filename, 'r') as f:
        header = f.readline().strip().split(" ")[2:]

    data = pd.read_csv(filename, comment='#', names=header,
                       delimiter='\s+', index_col=0)
    return data


class Sgoop:
    """
    Copying some handling from plumitas to deal with plumed files more easily.
    """
    def __init__(self, max_cal_traj, ct_col=None, rc=None, rc_bin=20, wells=2, d=1):
        self.max_cal_traj = read_plumed_file(max_cal_traj)
        self.ct_col = ct_col
        self.rc = rc
        self.rc_bin = rc_bin
        self.wells = wells
        self.d = d


def load(colvar: str, **kwargs):
    return Sgoop(colvar, **kwargs)
