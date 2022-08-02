import numpy as np


def stack_x_axis_times(times, num):
    """
    Stack x axis data(mostly time in EEG), used in bokeh.plotting.figure.multi_line
    :param times: x axis data, mostly time
    :param num: how many channels we want to use
    :return: stacked time steps
    """
    stacked_times = []
    for i in range(num):
        stacked_times.append(times)

    return stacked_times


def stack_y_axis_signals(signals, num, offset, *channels):
    """
    Stack y axis(mostly signal, time series), used in bokeh.plotting.figure.multi_line
    :param signals: signal time series we want to stack together
    :param num: how many channel we want to use
    :param offset: offset between each channel
    :param channels: selected channels
    :return: stacked signals
    """
    if channels:
        channels = channels[0]
    else:
        channels = np.arange(0, num, 1)

    stacked_signals = []
    for i in range(num):
        stacked_signals.append(signals[channels[i], :] + i * offset)

    return stacked_signals


def stackc_tick_loc(signal, num, offset):
    """
    Stacked tick location for the bokeh plot.
    :param signal: time series signal
    :param num: channel number
    :param offset: offset between different channels
    :return: stacked ticker
    """
    y_tick_loc = []
    for i in range(num):
        y_tick_loc.append(np.mean(signal[i, :]) + i * offset)

    y_tick_loc = np.around(y_tick_loc, 5)
    return y_tick_loc