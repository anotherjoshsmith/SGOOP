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

    def __init__(self, max_cal_colvar, metad_colvar=None, v_minus_c_col=None,
                 cv_cols= None, rc_bins=20, wells=2, d=1, rc_list=None,
                 prob_list=None, sg_list=None, ev_list=None):
        # read unbiased traj for max cal and metad_traj for probability
        self.max_cal_traj = read_plumed_file(max_cal_colvar)
        self.metad_traj = read_plumed_file(metad_colvar)
        self.v_minus_c_col = v_minus_c_col
        self.cv_cols = cv_cols

        # sgoop parameters
        self.rc_bins = rc_bins
        self.wells = wells
        self.d = d

        # rc value bin for max cal trajectory
        self.binned_rc_traj = None

        # probability for each rc bin from metad
        self.prob = None

        # storage dict for postprocessing
        self.storage_dict = {
            'rc_list': rc_list,
            'prob_list': prob_list,
            'ev_list': ev_list,
            'sg_list': sg_list
        }


def load(max_cal_colvar, metad_colvar=None, **kwargs):
    return Sgoop(max_cal_colvar, metad_colvar, **kwargs)
