from flask import render_template, Blueprint

import mne
import yaml
import numpy as np
from scipy import signal
from .source_utils import FrequencyAnalysis as fa
from .source_utils import RawDF, SpectrumDF, FrequencyAnalysis
from bokeh.embed import server_document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, TextInput
from bokeh.plotting import figure


def cfc_bkapp(doc):
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()

    # import os
    # print(os.listdir('./'))
    raw_edf = RawDF(filename=config['FILENAME'])
    current_channel = 0
    data = raw_edf.raw.get_data()[current_channel]
    ch_num = raw_edf.nchan
    fs = raw_edf.freq
    fNQ = fs / 2
    notch_freq = config['NOTCH']
    quality_factor = config['QUALITY']
    Wn1 = [config['WIN1'][0], config['WIN1'][1]]
    Wn2 = [config['WIN2'][0], config['WIN2'][1]]

    if notch_freq != None:
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
        data = signal.filtfilt(b_notch, a_notch, data)

    n = 100  # filter order,
    b1 = signal.firwin(n, Wn1, nyq=fNQ, pass_zero=False, window='hamming')
    b2 = signal.firwin(n, Wn2, nyq=fNQ, pass_zero=False, window='hamming')
    V1 = signal.filtfilt(b1, 1, data)
    V2 = signal.filtfilt(b2, 1, data)
    phi = np.angle(signal.hilbert(V1))
    amp = abs(signal.hilbert(V2))

    t = np.arange(0, len(data) / fs, 1 / fs)
    t_int = [24, 26]
    source1 = ColumnDataSource(data=dict(t=t[int(t_int[0] * fs):int(t_int[1] * fs)],
                                         signal=data[int(t_int[0] * fs):int(t_int[1] * fs)],
                                         V1=V1[int(t_int[0] * fs):int(t_int[1] * fs)],
                                         V2=V2[int(t_int[0] * fs):int(t_int[1] * fs)],
                                         ))

    p1 = figure(width=600, height=300)
    p1.line('t', 'signal', source=source1, line_color='skyblue', line_width=2, legend_label="data")
    p1.line('t', 'V1', source=source1, line_color='orange', line_width=2, legend_label="V1")
    p1.line('t', 'V2', source=source1, line_color='green', line_width=2, legend_label="V2")

    p1.legend.click_policy = 'hide'

    p_bins = np.arange(-np.pi, np.pi, 0.1)
    a_mean = np.zeros(np.size(p_bins) - 1)
    p_mean = np.zeros(np.size(p_bins) - 1)
    for k in range(np.size(p_bins) - 1):  # For each phase bin,
        pL = p_bins[k]  # ... lower phase limit,
        pR = p_bins[k + 1]  # ... upper phase limit.
        indices = (phi >= pL) & (phi < pR)  # Find phases falling in bin,
        a_mean[k] = np.mean(amp[indices])  # ... compute mean amplitude,
        p_mean[k] = np.mean([pL, pR])  # ... save center phase.


    source2 = ColumnDataSource(data=dict(V1=p_mean, V2=a_mean))
    p2 = figure(width=600, height=300)
    p2.line('V1', 'V2', source=source2, line_color='skyblue', line_width=2, )

    file_input1 = TextInput(title="Base File(Don't change): ", value=config['FILENAME'])
    file_input2 = TextInput(title='Compare File: ', value=config['FILENAME'])
    channel_slider1 = Slider(value=0, start=0, end=ch_num, step=1, width=400, title='Channel Base: ')
    channel_slider2 = Slider(value=0, start=0, end=ch_num, step=1, width=400, title='Channel Comparison: ')
    current_channel_name1 = TextInput(title='Current Base Channel:', value=raw_edf.ch_names[0])
    current_channel_name2 = TextInput(title='Current Comparison Channel:', value=raw_edf.ch_names[0])
    notch_input1 = TextInput(title='Notch Filter Base:', value='None')
    notch_input2 = TextInput(title='Notch Filter Comparison:', value='None')
    time_interval_input1 = TextInput(title='Time Interval Base(time point):', value='All')
    time_interval_input2 = TextInput(title='Time Interval Comparison(time point):', value='All')
    band_input1 = TextInput(title='Frequency Base: ', value='{wn1},{wn2}'.format(wn1=Wn1[0], wn2=Wn1[1]))
    band_input2 = TextInput(title='Frequency Comparison: ', value='{wn1},{wn2}'.format(wn1=Wn2[0], wn2=Wn2[1]))
    time_slider = Slider(value=0, start=0, end=len(data) / fs, title="Demo for 2s")
    item_selector1 = Select(title="V1 Option:", value="phase", options=["phase", "amplitude"])
    item_selector2 = Select(title="V2 Option:", value="amplitude", options=["phase", "amplitude"])

    file_bank = {config['FILENAME']: raw_edf}

    def update_data(attribute, old, new):
        file_update2 = file_input2.value
        channel_update1 = channel_slider1.value
        channel_update2 = channel_slider2.value
        current_channel_name1.value = raw_edf.ch_names[int(channel_update1)]
        current_channel_name2.value = raw_edf.ch_names[int(channel_update2)]
        notch_update1 = notch_input1.value
        notch_update2 = notch_input2.value
        time_interval_update1 = time_interval_input1.value
        time_interval_update2 = time_interval_input2.value
        band_update1 = band_input1.value
        band_update2 = band_input2.value
        time_demo_update = time_slider.value
        item_update1 = item_selector1.value
        item_update2 = item_selector2.value

        data_base = raw_edf.raw.get_data()[channel_update1]

        if file_update2 in file_bank.keys():
            raw_noba = file_bank[file_update2]
            data_noba = raw_noba.raw.get_data()[channel_update2]
        else:
            raw_noba = RawDF(filename=file_update2)
            data_noba = raw_noba.raw.get_data()[channel_update2]
            file_bank[file_update2] = raw_noba

        if time_interval_update1.lower() == 'all':
            y_base_filted = data_base
        else:
            interval_update1 = time_interval_update1.split(',')
            interval_update1[0] = int(interval_update1[0])
            interval_update1[1] = int(interval_update1[1])
            y_base_filted = data_base[interval_update1[0]:interval_update1[1]]

        if time_interval_update2.lower() == 'all':
            y_noba_filted = data_noba
        else:
            interval_update2 = time_interval_update2.split(',')
            interval_update2[0] = int(interval_update2[0])
            interval_update2[1] = int(interval_update2[1])
            y_noba_filted = data_noba[interval_update2[0]:interval_update2[1]]

        if notch_update1.lower() != 'none':
            try:
                cutoff = float(notch_update1)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.notch_filter(y_base_filted, cutoff, fs)
        else:
            pass

        if notch_update2.lower() != 'none':
            try:
                cutoff = float(notch_update2)
            except ValueError:
                print("Could not convert this input to float!")

            y_noba_filted = fa.notch_filter(y_noba_filted, cutoff, fs)
        else:
            pass

        band_update1 = band_update1.split(',')
        band_update2 = band_update2.split(',')
        Wn_update1 = [int(band_update1[0]), int(band_update1[1])]
        Wn_update2 = [int(band_update2[0]), int(band_update2[1])]

        b_update1 = signal.firwin(n, Wn_update1, nyq=fNQ, pass_zero=False, window='hamming')
        b_update2 = signal.firwin(n, Wn_update2, nyq=fNQ, pass_zero=False, window='hamming')
        V_update1 = signal.filtfilt(b_update1, 1, y_base_filted)
        V_update2 = signal.filtfilt(b_update2, 1, y_noba_filted)
        phi1 = np.angle(signal.hilbert(V_update1))
        phi2 = np.angle(signal.hilbert(V_update2))
        amp1 = abs(signal.hilbert(V_update1))
        amp2 = abs(signal.hilbert(V_update2))
        print(phi1)
        print(phi2)
        print(amp2)

        t_int_update = [int(time_demo_update), 2 + int(time_demo_update)]
        source1.data = dict(t=t[int(t_int_update[0] * fs):int(t_int_update[1] * fs)],
                            signal=data[int(t_int_update[0] * fs):int(t_int_update[1] * fs)],
                            V1=V1[int(t_int_update[0] * fs):int(t_int_update[1] * fs)],
                            V2=V2[int(t_int_update[0] * fs):int(t_int_update[1] * fs)],
                            )

        a_mean1 = np.zeros(np.size(p_bins) - 1)
        p_mean1 = np.zeros(np.size(p_bins) - 1)
        a_mean2 = np.zeros(np.size(p_bins) - 1)
        p_mean2 = np.zeros(np.size(p_bins) - 1)
        for k in range(np.size(p_bins) - 1):  # For each phase bin,
            pL = p_bins[k]  # ... lower phase limit,
            pR = p_bins[k + 1]  # ... upper phase limit.
            indices1 = (phi1 >= pL) & (phi1 < pR)  # Find phases falling in bin,
            a_mean1[k] = np.mean(amp1[indices1])  # ... compute mean amplitude,
            p_mean1[k] = np.mean([pL, pR])  # ... save center phase.

            indices2 = (phi1 >= pL) & (phi1 < pR)  # Find phases falling in bin,
            a_mean2[k] = np.mean(amp2[indices2])  # ... compute mean amplitude,
            p_mean2[k] = np.mean([pL, pR])  # ... save center phase.


        if item_update1 == 'phase':
            source2.data['V1'] = p_mean1
        elif item_update1 == 'amplitude':
            source2.data['V1'] = a_mean1

        if item_update2 == 'phase':
            source2.data['V2'] = p_mean2
        elif item_update2 == 'amplitude':
            source2.data['V2'] = a_mean2

    for w in [file_input2, notch_input1, notch_input2, time_interval_input1,
              time_interval_input2, band_input1, band_input2, item_selector1, item_selector2]:
        w.on_change('value', update_data)

    for w in [channel_slider1, channel_slider2, time_slider]:
        w.on_change('value_throttled', update_data)

    left_widgets = column(file_input1, channel_slider1, current_channel_name1, time_interval_input1, band_input1, item_selector1, notch_input1)
    right_widgets = column(file_input2, channel_slider2, current_channel_name2, time_interval_input2, band_input2, item_selector2,
                           notch_input2)

    doc.add_root(row(column(left_widgets, p1, time_slider), column(right_widgets, p2)))


bp = Blueprint("cfc checker", __name__, url_prefix="/cfc")


@bp.route("/", methods=["GET"])
def bkapp_page():
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    script = server_document("http://localhost:5006/cfc_bkapp")
    return render_template("cfc.html", script=script, template="Flask", port=config['PORT'])

# def bk_worker():
#     server = Server({'/spectrum_bkapp': spectrum_bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
#     server.start()
#     server.io_loop.start()
#
#
# Thread(target=bk_worker).start()
