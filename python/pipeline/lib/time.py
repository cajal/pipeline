from pipeline import PipelineException
import numpy as np
from scipy.interpolate import interp1d

def ts2sec(ts, packetLen=0):
    """
    Convert 10MHz timestamps from Saumil's patching program (ts) to seconds (s)

    :param ts: timestamps
    :param packetLen: length of timestamped packets
    :returns:
        timestamps converted to seconds
        system time (in seconds) of t=0
        bad camera indices from 2^31:2^32 in camera timestamps prior to 4/10/13
    """
    ts = ts.astype(float)

    # find bad indices in camera timestamps and replace with linear est
    bad_idx = ts ==2**31-1
    if sum(bad_idx) > 10:
        raise PipelineException('Bad camera ts...')
        x = np.where(~bad_idx)[0]
        x_bad = np.where(bad_idx)[0]
        f = interp1d(x, ts[~bad_idx],'linear')
        ts[bad_idx] = f(x_bad)

    #  remove wraparound
    wrapInd = np.where(np.diff(ts)<0)[0]
    while not len(wrapInd) == 0:
        ts[wrapInd[0]+1:] += 2**32
        wrapInd = np.where(np.diff(ts)<0)[0]

    s = ts/1e7

    # Remove offset, and if not monotonically increasing (i.e. for packeted ts), interpolate
    if np.any(np.diff[s]<=0):
        # Check to make sure it's packets
        diffs = np.where(np.diff(s)>0)[0]
        assert packetLen == diffs[0]

        # Subtract duration of first packet from all timestamps
        packetDur = s[packetLen-1]-s[0]
        s -= packetDur

        # Interpolate
        not_zero = np.hstack((0, diffs+1))
        f = interp1d(not_zero, s[not_zero],'linear')
        s = f(np.arange(len(s)))
    s -= s[0]
