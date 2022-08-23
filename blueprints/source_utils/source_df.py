import numpy as np
import mne
import pandas as pd
from scipy import signal
from collections import OrderedDict
from .frequency_analysis import FrequencyAnalysis
import re

"""Need to implement notch_policy=False"""


class RawDF():
    """
    read file and save in this object
    raw: raw variable from mne
    ch_names: channel names
    nchan: channel number
    data: save the data into a Dataframe, each column is a channel
    brain_regions: generated brain regions using channel names
    dict_df_raw: a dictionary, save each brain region data in a key
    """

    def __init__(self, **kwargs):
        """
        initial method
        :param kwargs:
            filename: must exist, generate object using this file
            channels_policy: all or part stating whether to use whole data set
        """
        if "filename" not in kwargs.keys():
            raise ValueError("Must declare the file path")
        self.raw = mne.io.read_raw_edf(kwargs['filename'])
        self.freq = self.raw.info['sfreq']

        if 'channels_policy' in kwargs.keys():
            self.channels_policy = kwargs['channels_policy']
            if self.channels_policy == 'all':
                self.ch_names = self.raw.ch_names
                self.nchan = self.raw.info['nchan']
            elif self.channels_policy == 'customize':
                if "ch_names" in kwargs.keys():
                    self.ch_names = kwargs['ch_names']
        else:
            self.nchan = self.raw.info['nchan']
            self.ch_names = self.raw.ch_names

        self.data = pd.DataFrame(data=self.raw.get_data().T, columns=self.ch_names)
        self.brain_regions = self.find_brain_region()
        self.dict_df_raw = self.df_to_dict_raw()

        """
        ['acq_pars',
         'acq_stim',
         'ctf_head_t',
         'description',
         'dev_ctf_t',
         'dig',
         'experimenter',
         'utc_offset',
         'device_info',
         'file_id',
         'highpass',
         'hpi_subsystem',
         'kit_system_id',
         'helium_info',
         'line_freq',
         'lowpass',
         'meas_date',
         'meas_id',
         'proj_id',
         'proj_name',
         'subject_info',
         'xplotter_layout',
         'gantry_angle',
         'bads',
         'chs',
         'comps',
         'events',
         'hpi_meas',
         'hpi_results',
         'projs',
         'proc_history',
         'custom_ref_applied',
         'sfreq',
         'dev_head_t',
         'ch_names',
         'nchan']
        """

    def find_brain_region(self):
        """
        Split every channel names, extract the string at the beginning.
        :return: channel kinds
        """
        splitted_ch_names = [re.split(r'(\d+)', i) for i in self.ch_names]
        channel_split_string_num = np.array([i[0] for i in splitted_ch_names])
        self.brain_regions = np.unique(channel_split_string_num)
        return self.brain_regions

    def df_to_dict_raw(self):
        """
        turn the full dataframe to specific brain region dataframe and dave to a dictionary
        :return: splited dictionary
        """
        dict_df_regions_signals = OrderedDict()
        for region in self.brain_regions:
            den_mask = [ch_name.startswith(region) for ch_name in self.ch_names]
            dict_df_regions_signals[region] = self.data.loc[:, den_mask]

        return dict_df_regions_signals


    def filter_data_from_region(self, region: list):
        data = self.raw.get_data()
        data_mask = np.zeros(self.nchan)
        for i in region:
            for j in range(len(self.ch_names)):
                if self.ch_names[j].startswith(i):
                    data_mask[j] = 1

        data_mask = data_mask.astype('bool')
        data_masked = data[data_mask]
        chan_masked = np.array(self.ch_names)[data_mask]
        return [chan_masked, data_masked]


class SpectrumDF(RawDF):
    """

    """

    def __init__(self, **kwargs):
        super(SpectrumDF, self).__init__(**kwargs)

        self.lowpass_freq = kwargs['lowpass_freq'] if 'lowpass_freq' in kwargs.keys() else None
        self.highpass_freq = kwargs['highpass_freq'] if 'highpass_freq' in kwargs.keys() else None
        self.notch_freq = kwargs['notch_freq'] if 'notch_freq' in kwargs.keys() else 60
        self.nperseg = kwargs['nperseg'] if 'nperseg' in kwargs.keys() else 5120
        self.quality_factor = kwargs['quality_factor'] if 'quality_factor' in kwargs.keys() else 30
        self.order = kwargs['order'] if 'order' in kwargs.keys() else 3
        self.log = kwargs['log'] if 'log' in kwargs.keys() else True

        self.data_filted = self.filt_signal()
        self.fseq, self.den = self.data_filted_to_den()
        self.dict_df_fft = self.df_to_dict_fft()

        # self.fs, self.den, self.df = self.compute_spectrum()
        # self.dict_df_fft = self.df_to_dict_fft()

    def filt_signal(self):
        data_filted = self.data
        if self.notch_freq:
            data_filted = FrequencyAnalysis.notch_filter(data_filted, self.notch_freq,
                                                         self.freq, self.quality_factor)
        if isinstance(data_filted, np.ndarray):
            data_filted = pd.DataFrame(data=data_filted, columns=self.ch_names)

        if self.highpass_freq:
            data_filted = FrequencyAnalysis.butter_highpass_filter(data_filted, self.highpass_freq,
                                                                   self.freq, self.order)
        if isinstance(data_filted, np.ndarray):
            data_filted = pd.DataFrame(data=data_filted, columns=self.ch_names)
        return data_filted

    def data_filted_to_den(self):
        den = np.zeros((int(self.nperseg / 2 + 1), self.data_filted.shape[1]))
        for i in range(self.data_filted.shape[1]):
            fseq, den[:, i] = signal.welch(self.data_filted.iloc[:, i], self.freq, nperseg=self.nperseg)

        if self.log:
            np.seterr(divide='ignore')
            den = np.log10(den)
            np.seterr(divide='warn')

        den = pd.DataFrame(data=den, columns=self.ch_names)
        return [fseq, den]

    def compute_spectrum(self, nperseg=5120, time_window_policy='all', time_window=None,
                         notch_policy=True, notch_freq=60):
        """

        :param freq: sampling frequency of the signal
        :param nperseg: Length of each segment, (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
        :return:
        """

        freq = int(self.raw.info['sfreq'])
        if time_window_policy == 'all':
            if notch_policy == True:
                b_notch, a_notch = signal.iirnotch(notch_freq, 30, freq)
                den = np.zeros((self.nchan, int(nperseg / 2 + 1)))
                for i in range(self.nchan):
                    y_notched = signal.filtfilt(b_notch, a_notch, self.data.iloc[:, i])
                    fs, den[i] = signal.welch(y_notched, freq, nperseg=nperseg)

                fs = np.tile(fs, (self.nchan, 1))

                # self.df = {"fs": pd.DataFrame(data=fs.T, columns=self.ch_names),
                #            "den": pd.DataFrame(data=den.T, columns=self.ch_names)}
                columns = np.hstack((np.array(self.ch_names), np.array(["x_" + i for i in self.ch_names])))

                df = pd.DataFrame(data=np.hstack((den.T, fs.T)), columns=columns)
                fs = pd.DataFrame(data=fs.T, columns=self.ch_names)
                den = pd.DataFrame(data=den.T, columns=self.ch_names)

                return [fs, den, df]
            elif notch_policy == False:
                pass

        elif time_window_policy == 'part':
            b_notch, a_notch = signal.iirnotch(60, 30, freq)
            den = np.zeros((self.nchan, int(nperseg / 2 + 1)))
            for i in range(self.nchan):
                y_notched = signal.filtfilt(b_notch, a_notch, self.data[:, time_window[0]:time_window[1]][i])
                fs, den[i] = signal.welch(y_notched, freq, nperseg=nperseg)

            fs = np.tile(fs, (self.nchan, 1))
            fs = fs
            den = den

            # self.df = {"fs": pd.DataFrame(data=fs.T, columns=self.ch_names),
            #            "den": pd.DataFrame(data=den.T, columns=self.ch_names)}
            columns = np.hstack((np.array(self.ch_names), np.array(["x_" + i for i in self.ch_names])))

            df = pd.DataFrame(data=np.hstack((den.T, fs.T)), columns=columns)
            fs = pd.DataFrame(data=fs.T, columns=self.ch_names)
            den = pd.DataFrame(data=den.T, columns=self.ch_names)
            return [fs, den, df]

        else:
            raise ValueError("time window policy has to be either 'all' or 'part'!")

    def df_to_dict_fft(self):
        """
        turn the full dataframe to specific brain region dataframe and dave to a dictionary
        :return: splited dictionary
        """
        dict_df_regions_signals = {}
        for region in self.brain_regions:
            den_mask = [ch_name.startswith(region) for ch_name in self.ch_names]
            dict_df_regions_signals[region] = self.den.loc[:, den_mask]

        return dict_df_regions_signals

# if __name__ == '__main__':
#     from frequency_analysis import FrequencyAnalysis as fa
#     import matplotlib.pyplot as plt
#     df = RawDF(filename='E:/Subject_4_event_7_base.edf')
#     data = df.raw.get_data()[:10]
#     tmp = fa.butter_highpass_filter(data, 1, 1024)
#     tmp1 = fa.notch_filter(tmp, 60, 1024)
#     fs, den = fa.data_filted_to_den(tmp1, 2561, 1024)





