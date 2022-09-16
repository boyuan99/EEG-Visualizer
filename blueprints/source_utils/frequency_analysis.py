import mne
import numpy as np
import pandas as pd
import scipy
from scipy import fft, signal, integrate


class FrequencyAnalysis():
    @classmethod
    def butter_bandpass_filter(cls, data, lowcut, highcut, fs, order=5):
        '''
        Band pass filter based on scipy.signal.butter
        :param data: 1-channel data
        :param lowcut: lower cutoff
        :param highcut: higher cutoff
        :param fs: signal frequency
        :param order: parameter in the butter function
        :return: filtered data
        '''
        lowcut = 2 * lowcut / fs
        highcut= 2 * highcut/fs
        b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
        y = signal.filtfilt(b, a, data)
        return y

    @classmethod
    def butter_lowpass_filter(cls, data, cutoff, fs, order=5):
        '''
        Low pass filter based on scipy.signal.butter
        :param data: 1-channel signal
        :param cutoff: cutoff frequency
        :param fs: signal frequency
        :param order: parameter for the butter function
        :return: filtered signal
        '''
        normal_cutoff = 2 * cutoff / fs
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        if isinstance(data, pd.DataFrame):
            y = np.zeros((data.shape[0], data.shape[1]))
            for i in range(data.shape[1]):
                y[:, i] = signal.filtfilt(b, a, data.iloc[:, i])

            y = pd.DataFrame(data=y, columns=data.columns)

        elif isinstance(data, np.ndarray):
            if len(data.shape)>1:
                y = np.zeros((data.shape[0], data.shape[1]))
                for i in range(data.shape[0]):
                    y[i] = signal.filtfilt(b, a, data[i])
            else:
                y = signal.filtfilt(b, a, data)

        return y

    @classmethod
    def butter_highpass_filter(cls, data, cutoff, fs, order=5):
        '''
        High pass filter based on scipy.signal.butter
        :param data: 1-channel signal
        :param cutoff: cutoff frequency
        :param fs: signal sampling frequency
        :param order: order parameter for signal.signal.butter function
        :return: filtered signal
        '''
        normal_cutoff = 2*cutoff/fs
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

        if isinstance(data, pd.DataFrame):
            y = np.zeros((data.shape[0], data.shape[1]))
            for i in range(data.shape[1]):
                y[:, i] = signal.filtfilt(b, a, data.iloc[:, i])

            y = pd.DataFrame(data=y, columns=data.columns)

        elif isinstance(data, np.ndarray):
            if len(data.shape) > 1:
                y = np.zeros((data.shape[0], data.shape[1]))
                for i in range(data.shape[0]):
                    y[i] = signal.filtfilt(b, a, data[i])
            else:
                y = signal.filtfilt(b, a, data)

        return y

    @classmethod
    def notch_filter(cls, data, notch_freq, smp_fs, quality_factor=30):
        '''
        Notch filter based on scipy.signal.iirnotch
        :param data: 1-channel signal
        :param notch_freq: notch frequency
        :param fs: signal sampling frequency
        :param quality_factor: parameter for the scipy.signal.iirnotch function
        :return: filtered signal
        '''
        # if isinstance(data, pd.DataFrame):
        #     b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, smp_fs)
        #     den = np.zeros((int(nperseg / 2 + 1), data.shape[1]))
        #     for i in range(data.shape[1]):
        #         y_notched = signal.filtfilt(b_notch, a_notch, data.iloc[:, i])
        #         fs, den[:, i] = signal.welch(y_notched, smp_fs, nperseg=nperseg)

        if isinstance(data, pd.DataFrame):
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, smp_fs)
            y_notched = np.zeros((data.shape[0], data.shape[1]))
            for i in range(data.shape[1]):
                y_notched[:, i] = signal.filtfilt(b_notch, a_notch, data.iloc[:, i])

            y_notched = pd.DataFrame(data=y_notched, columns=data.columns)

        elif isinstance(data, np.ndarray):
            b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, smp_fs)
            if len(data.shape) > 1:
                y_notched = np.zeros((data.shape[0], data.shape[1]))
                for i in range(data.shape[0]):
                    y_notched[i] = signal.filtfilt(b_notch, a_notch, data[i])
            else:
                y_notched = signal.filtfilt(b_notch, a_notch, data)

        return y_notched

    @classmethod
    def data_filted_to_den(cls, data, len, smp_freq, nperseg=5120, log=True):
        """
        turn raw data to frequency domain
        :param data: input data, usually pd.DataFrame
        :param len: the length of the output data, nperseg/2+1
        :param smp_freq: sampling frequency of the raw data
        :param nperseg: nperseg in welch method
        :param log: if apply log function to the output data
        :return: [fs: corresponding frequency, 1d-array,
                  den: power density, pd.DataFrame]
        """
        if isinstance(data, pd.DataFrame):
            den = np.zeros((len, data.shape[1]))
            for i in range(data.shape[1]):
                fseq, den[:, i] = signal.welch(data.iloc[:, i], smp_freq, nperseg=nperseg)

            if log:
                np.seterr(divide='ignore')
                den = np.log10(den)
                np.seterr(divide='warn')

            den = pd.DataFrame(data=den, columns=data.columns)
            return [fseq, den]

        if isinstance(data, np.ndarray):
            den = np.zeros((data.shape[0], len))
            for i in range(data.shape[0]):
                fseq, den[i] = signal.welch(data[i], smp_freq, nperseg=nperseg)

            if log:
                np.seterr(divide='ignore')
                den = np.log10(den)
                np.seterr(divide='warn')

            return [fseq, den]

    @classmethod
    def signal_time_in_freq_out(cls, data, cutoff, fs, filter_type, order=5):
        '''
        Input the signal in time domain, and this function will output the processed frequency domain information

        :param data: Input data (signal, time series, 1-channel)
        :param cutoff: For notch, low and pass, cutoff is a number, for band, it's a list like [low, high]
        :param fs: The input signal frequency
        :param filter_type: lowpass, highpass, notch, or bandpass
        :param order: The order for scipy.signal.butter function
        :return: [xf_half, yf_half]: Turn the filtered data into frequency domain and only output half
                                    (positive frequency) of the frequency domain signal
        '''
        if filter_type == 'low':
            y = cls.butter_lowpass_filter(data, cutoff, fs, order)
            yf = fft.fft(y)
            xf = fft.fftfreq(len(y), 1 / fs)

            xf_ind = np.where(xf >= 0)
            yf_half = yf[xf_ind]
            xf_half = xf[xf_ind]
            return [xf_half, yf_half]

        elif filter_type == 'high':
            y = cls.butter_highpass_filter(data, cutoff, fs, order)
            yf = fft.fft(y)
            xf = fft.fftfreq(len(y), 1 / fs)

            xf_ind = np.where(xf >= 0)
            yf_half = yf[xf_ind]
            xf_half = xf[xf_ind]
            return [xf_half, yf_half]

        elif filter_type == 'band':
            lowcut = cutoff[0]
            highcut = cutoff[1]
            y = cls.butter_bandpass_filter(data, lowcut, highcut, fs, order)
            yf = fft.fft(y)
            xf = fft.fftfreq(len(y), 1 / fs)

            xf_ind = np.where(xf >= 0)
            yf_half = yf[xf_ind]
            xf_half = xf[xf_ind]
            return [xf_half, yf_half]

        elif filter_type == 'notch':
            # Notice that the order here is the quality factor for notch
            y = cls.notch_filter(data, cutoff, fs, order)
            yf = fft.fft(y)
            xf = fft.fftfreq(len(y), 1 / fs)

            xf_ind = np.where(xf >= 0)
            yf_half = yf[xf_ind]
            xf_half = xf[xf_ind]
            return [xf_half, yf_half]

    @classmethod
    def db4_filter(cls, data, freq):
        '''
        Apply db4 filter on EEG data
        :param data: 1-channel data, a time series
        :param freq:
        :return:
        '''

        xf_60, yf_60 = cls.signal_time_in_freq_out(data, 60, freq, 'notch', order=30)
        data = cls.notch_filter(data, 60, freq, quality_factor=30)

        xf_0_64, yf_0_64 = cls.signal_time_in_freq_out(data, 64, freq, 'low')
        y_0_64 = cls.butter_lowpass_filter(data, 64, freq)

        xf_0_32, yf_0_32 = cls.signal_time_in_freq_out(y_0_64, 32, freq, 'low')
        y_0_32 = cls.butter_lowpass_filter(y_0_64, 32, freq)

        xf_32_64, yf_32_64 = cls.signal_time_in_freq_out(y_0_64, 32, freq, 'high')
        y_32_64 = cls.butter_highpass_filter(y_0_64, 32, freq)
        f_32_64, power_den_32_64 = signal.welch(y_32_64, freq, nperseg=1024)
        power_32_64 = integrate.trapz(power_den_32_64, f_32_64)

        xf_32_64_tmp, yf_32_64_tmp = cls.signal_time_in_freq_out(data, [32, 64], freq, 'band')
        y_32_64_tmp = cls.butter_bandpass_filter(data, 32, 64, freq)

        xf_0_16, yf_0_16 = cls.signal_time_in_freq_out(y_0_32, 16, freq, 'low')
        y_0_16 = cls.butter_lowpass_filter(y_0_32, 16, freq)

        xf_16_32, yf_16_32 = cls.signal_time_in_freq_out(y_0_32, 16, freq, 'high')
        y_16_32 = cls.butter_highpass_filter(y_0_32, 16, freq)
        f_16_32, power_den_16_32 = signal.welch(y_16_32, freq, nperseg=1024)
        power_16_32 = integrate.trapz(power_den_16_32, f_16_32)

        xf_0_8, yf_0_8 = cls.signal_time_in_freq_out(y_0_16, 8, freq, 'low')
        y_0_8 = cls.butter_lowpass_filter(y_0_16, 8, freq)

        xf_8_16, yf_8_16 = cls.signal_time_in_freq_out(y_0_16, 8, freq, 'high')
        y_8_16 = cls.butter_highpass_filter(y_0_16, 8, freq)
        f_8_16, power_den_8_16 = signal.welch(y_8_16, freq, nperseg=1024)
        power_8_16 = integrate.trapz(power_den_8_16, f_8_16)

        xf_0_4, yf_0_4 = cls.signal_time_in_freq_out(y_0_8, 4, freq, 'low')
        y_0_4 = cls.butter_lowpass_filter(y_0_8, 4, freq)

        xf_4_8, yf_4_8 = cls.signal_time_in_freq_out(y_0_8, 4, freq, 'high')
        y_4_8 = cls.butter_highpass_filter(y_0_8, 4, freq)
        f_4_8, power_den_4_8 = signal.welch(y_4_8, freq, nperseg=1024)
        power_4_8 = integrate.trapz(power_den_4_8, f_4_8)

        xf_0_4, yf_0_4 = cls.signal_time_in_freq_out(y_0_4, 0.5, freq, 'high')
        y_0_4 = cls.butter_highpass_filter(y_0_8, 0.5, freq)
        f_0_4, power_den_0_4 = signal.welch(y_0_4, freq, nperseg=1024)
        power_0_4 = integrate.trapz(power_den_0_4, f_0_4)

        return [power_32_64, power_16_32, power_8_16, power_4_8, power_0_4]
