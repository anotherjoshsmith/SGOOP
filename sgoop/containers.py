import os.path as op
import pandas as pd


def read_plumed_file(filename):
    """
    Read PLUMED output files into pandas DataFrame, with colvars/bias column names
    and time indices.

    :param filename:
    :return: pd.DataFrame
    """
    if filename is None:
        return None

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
    def __init__(self, max_cal_colvar, metad_colvar=None, rc=None, num_rc_bins=20, wells=2, d=1):
        self.max_cal_traj = read_plumed_file(max_cal_colvar)
        self.metad_traj = read_plumed_file(metad_colvar)
        self.rc = rc
        self.num_rc_bins = num_rc_bins
        self.wells = wells
        self.d = d


def load(max_cal_colvar, metad_colvar=None, **kwargs):
    return Sgoop(max_cal_colvar, metad_colvar, **kwargs)
