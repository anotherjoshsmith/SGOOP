import os.path as op
import pandas as pd


def read_plumed_file(filename):
    """
    Read PLUMED output files into pandas DataFrame.

    Column names are parsed from the header of the Plumed file (e.g. COLVAR or HILLS)
    and indices are taken from the time column of the Plumed file.

    Parameters
    ----------
    filename : string
        Name of the plumed file that contains collective variable data
        (e.g. HILLS or COLVAR)

    Returns
    -------
    data : pd.DataFrame
        Pandas DataFrame with column names parsed from CV/bias labels in the plumed file
        header and time column used for the index.

    Examples
    --------
    Read COLVAR file from a MetaD simulation into a DataFrame named metad_traj.

    >>> metad_traj = read_plumed_file('COLVAR')
    >>> metad_traj.head()
    # todo: finish example with output
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
    """
        Load SGOOP object from Plumed output.

        Parameters
        ----------
        max_cal_colvar : string
            Name of the plumed CV file (e.g. COLVAR) from unbiased simulation, to be
            used for MaxCal calculation.
        metad_colvar : string, None
            Name of the plumed CV file (e.g. COLVAR) from MetaD biased simulation, to
            be used for potential of mean force calculation.
        **kwargs : dict
            Keyword arguments associated with the parameters for SGOOP analysis.

        Returns
        -------
        Sgoop : object
            Instance of the Sgoop class, designed to store the data and parameters to
            be used for the SGOOP calculation.

        Examples
        --------
        # todo: finish example with output
        """
    return Sgoop(max_cal_colvar, metad_colvar, **kwargs)
