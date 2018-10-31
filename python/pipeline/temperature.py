import datajoint as dj
from pipeline import experiment
from commons import lab
from datajoint.jobs import key_hash
import os
import numpy as np

from .utils import h5, signal
from .exceptions import PipelineException
from . import notify


schema = dj.schema('pipeline_temperature')


@schema
class Temperature(dj.Imported):
    definition = """ # temperature across the scan

    -> experiment.Scan
    ---
    temp_time                   : longblob      # (secs) times of each temperature sample in behavior clock
    temperatures                : longblob      # (Celsius) temperature trace
    median_temperature          : float         # (Celsius) median temperature over the recording
    temp_ts=CURRENT_TIMESTAMP   : timestamp
    """

    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.BehaviorFile().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Get counter timestamps and convert to seconds
        ts = h5.ts2sec(data['ts'], is_packeted=True)

        # Read temperature (if available) and invalidate points with unreliable timestamps
        temp_raw = data.get('temperature', None)
        if temp_raw is None:
            raise PipelineException('Scan {animal_id}-{session}-{scan_idx} does not have '
                                    'temperature data'.format(**key))
        temp_raw[np.isnan(ts)] = float('nan')

        # Read temperature and smooth it
        temp_celsius = (temp_raw * 100 - 32) / 1.8  # F to C
        sampling_rate = int(round(1 / np.nanmedian(np.diff(ts))))  # samples per second
        smooth_temp = signal.low_pass_filter(temp_celsius, sampling_rate, cutoff_freq=1,
                                             filter_size=2 * sampling_rate)

        # Resample at 1 Hz
        downsampled_ts = ts[::sampling_rate]
        downsampled_temp = smooth_temp[::sampling_rate]

        # Insert
        self.insert1({**key, 'temp_time': downsampled_ts,
                      'temperatures': downsampled_temp,
                      'median_temperature': np.nanmedian(downsampled_temp)})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        ts, temperatures = (self & key).fetch1('temp_time', 'temperatures')

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts, temperatures)
        plt.ylabel('Temperature (C)')
        plt.xlabel('Seconds')
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'temperature for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def session_plot(self):
        """ Do a plot of how temperature progress through a session"""
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Check that plot is restricted to a single session
        session_key = self.fetch('KEY', limit=1)[0]
        session_key.pop('scan_idx')
        if len(self & session_key) != len(self):
            raise PipelineException('Plot can only be generated for one session at a '
                                    'time')

        # Get times and timestamps, scan_ts
        scan_indices, ts, temperatures = self.fetch('scan_idx', 'temp_time',
                                                    'temperatures', order_by='scan_idx')
        session_ts = (experiment.Session() & self).fetch1('session_ts')
        scan_ts = (experiment.Scan() & self).fetch('scan_ts', order_by='scan_idx')
        abs_ts = [(sts - session_ts).seconds + (t - t[0]) for sts, t in zip(scan_ts, ts)]

        # Plot
        fig = plt.figure(figsize=(10, 5))
        for abs_ts_, temp_, scan_idx in zip(abs_ts, temperatures, scan_indices):
            plt.plot(abs_ts_ / 3600, temp_, label='Scan {}'.format(scan_idx))  # in hours
        plt.title('Temperature for {animal_id}-{session} starting at {session_ts}'.format(
            session_ts=session_ts, **session_key))
        plt.ylabel('Temperature (Celsius)')
        plt.xlabel('Hour')
        plt.legend()

        # Plot formatting
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        plt.grid(linestyle='--', alpha=0.8)

        return fig


@schema
class TempDrift(dj.Computed):
    definition = """ # assuming temperature increases/decreases consistently, compute rate of change
    
    -> Temperature
    ---
    temp_slope              : float            # (C/hour) change in temperature
    rmse                    : float            # (C) root mean squared error of the fit
    """

    def _make_tuples(self, key):
        # Get all times and temperatures
        ts, temperatures = (Temperature() & key).fetch1('temp_time', 'temperatures')
        ts = ts[~np.isnan(temperatures)]
        temperatures = temperatures[~np.isnan(temperatures)]

        # Fit a line (robust regression)
        from sklearn import linear_model
        X = ts.reshape(-1, 1)
        y = temperatures
        model = linear_model.TheilSenRegressor()
        model.fit(X, y)

        #  Get results
        z_slope = model.coef_[0] * 3600 # C/hour
        rmse = np.sqrt(np.mean((temperatures - model.predict(X)) ** 2))

        self.insert1({**key, 'temp_slope': z_slope, 'rmse': rmse})