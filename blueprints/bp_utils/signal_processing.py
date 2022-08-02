import numpy as np
from scipy import fft, signal, integrate


def convolve_multichannel(signal, kernel, axis):
    """
    Apply 1D convolution for multichannel time series
    :param signal: multichannel signal
    :param kernel: kernel that will be applied to each channel
    :param axis: which axis we want to apply the convolution
    :return: convolved multichannel signal
    """
    if kernel[0] == -1:
        data_convolved = signal
    else:
        ch_num = signal.shape[axis]
        data_convolved = np.zeros_like(signal)
        for i in range(ch_num):
            data_convolved[i] = np.convolve(signal[i], kernel, mode='same')

    return data_convolved


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
    b, a = signal.butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass_filter(cls, data, cutoff, fs, order=5):
    '''
    Low pass filter based on scipy.signal.butter
    :param data: 1-channel signal
    :param cutoff: cutoff frequency
    :param fs: signal frequency
    :param order: parameter for the butter function
    :return: filtered signal
    '''
    b, a = signal.butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_highpass_filter(cls, data, cutoff, fs, order=5):
    '''
    High pass filter based on scipy.signal.butter
    :param data: 1-channel signal
    :param cutoff: cutoff frequency
    :param fs: signal sampling frequency
    :param order: order parameter for signal.signal.butter function
    :return: filtered signal
    '''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def notch_filter(cls, data, notch_freq, fs, quality_factor=30):
    '''
    Notch filter based on scipy.signal.iirnotch
    :param data: 1-channel signal
    :param notch_freq: notch frequency
    :param fs: signal sampling frequency
    :param quality_factor: parameter for the scipy.signal.iirnotch function
    :return: filtered signal
    '''
    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    y = signal.filtfilt(b, a, data)
    return y


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
