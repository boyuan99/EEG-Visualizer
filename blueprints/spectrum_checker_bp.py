from flask import render_template, Blueprint

import mne
import yaml
import numpy as np
from scipy import signal
from .source_utils import FrequencyAnalysis as fa
from .source_utils import RawDF, SpectrumDF, FrequencyAnalysis
from bokeh.embed import server_document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, HoverTool, TextInput
from bokeh.plotting import figure


def spectrum_bkapp(doc):
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()

    log_policy = config['LOG']
    lowpass_freq = config['LOWPASS']
    highpass_freq = config['HIGHPASS']
    notch_freq = config['NOTCH']
    nperseg = config['NPERSEG']
    quality_factor = config['QUALITY']
    order = config['ORDER']

    raw_edf = SpectrumDF(filename=config['FILENAME'],
                         lowpass_freq=lowpass_freq,
                         highpass_freq=highpass_freq,
                         notch_freq=notch_freq,
                         nperseg=nperseg,
                         quality_factor=quality_factor,
                         order=order,
                         log=log_policy,)

    fs = raw_edf.raw.info['sfreq']
    ch_num = raw_edf.nchan

    fseq = raw_edf.fseq
    den = raw_edf.den
    den_array = den.to_numpy().T

    source = ColumnDataSource(
        data=dict(
            x_base=fseq,
            y_base=den_array[0],
            x_noba=fseq,
            y_noba=den_array[0]
        )
    )
    hover = HoverTool(
        tooltips=[
            ("(x,y)", "($x, $y)"),
        ]
    )
    p = figure(height=400, width=900, x_range=(-5, 100))
    p.add_tools(hover)
    p.line('x_base', 'y_base', source=source, line_width=3, line_color='skyblue', legend_label='base')
    p.line('x_noba', 'y_noba', source=source, line_width=3, line_color='orange', legend_label='compare')
    p.legend.click_policy = 'hide'

    highpass_input = TextInput(title='Highpass Filter:', value='None')
    lowpass_input = TextInput(title='Lowpass Filter:', value='None')
    notch_input = TextInput(title='Notch Filter:', value='None')
    interval_input = TextInput(title='Time Interval:', value='All')
    file_input = TextInput(title='Compare File:', value='Default')
    channel_slider = Slider(value=0, start=0, end=ch_num, step=1, width=900, title='Channel')
    current_channel_name = TextInput(title='Current Channel:', value=raw_edf.ch_names[0])

    file_bank = dict()

    def update_data(attribute, old, new):
        file_update = file_input.value
        highpass_update = highpass_input.value
        lowpass_update = lowpass_input.value
        notch_update = notch_input.value
        interval_update = interval_input.value
        channel_update = channel_slider.value

        if file_update.lower() == "default":
            raw_base = raw_edf.raw
            data_base = raw_base.get_data()
            raw_noba = raw_edf.raw
            data_noba = raw_noba.get_data()

        else:
            if file_update in file_bank.keys():
                raw_base = raw_edf.raw
                data_base = raw_base.get_data()
                raw_noba, data_noba = file_bank[file_update]
            else:
                raw_noba_update = RawDF(filename=file_update)
                raw_base = raw_edf.raw
                data_base = raw_base.get_data()
                raw_noba = raw_noba_update.raw
                data_noba = raw_noba.get_data()
                file_bank[file_update] = [raw_noba, data_noba]

        current_channel_name.value = raw_edf.ch_names[channel_update]
        if interval_update.lower() == 'all':
            y_base_filted = data_base[channel_update]
            y_noba_filted = data_noba[channel_update]
        else:
            interval_update = interval_update.split(',')
            interval_update[0] = int(interval_update[0])
            interval_update[1] = int(interval_update[1])
            y_base_filted = data_base[channel_update, interval_update[0]:interval_update[1]+1]
            y_noba_filted = data_noba[channel_update, interval_update[0]:interval_update[1]+1]

        if highpass_update.lower() != 'none':
            try:
                cutoff = float(highpass_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.butter_highpass_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.butter_highpass_filter(y_noba_filted, cutoff, raw_edf.freq)

        else:
            pass

        if lowpass_update.lower() != 'none':
            try:
                cutoff = float(lowpass_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.butter_lowpass_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.butter_lowpass_filter(y_noba_filted, cutoff, raw_edf.freq)
        else:
            pass

        if notch_update.lower() != 'none':
            try:
                cutoff = float(notch_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.notch_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.notch_filter(y_noba_filted, cutoff, raw_edf.freq)
        else:
            pass

        f_base, den_base = signal.welch(y_base_filted, fs, nperseg=nperseg)
        f_noba, den_noba = signal.welch(y_noba_filted, fs, nperseg=nperseg)

        np.seterr(divide='ignore')
        den_base = np.log10(den_base)
        den_noba = np.log10(den_noba)
        np.seterr(divide='warn')

        source.data = dict(
            x_base=f_base,
            y_base=den_base,
            x_noba=f_noba,
            y_noba=den_noba
        )

    for w in [file_input, interval_input, highpass_input, lowpass_input, notch_input]:
        w.on_change('value', update_data)
    channel_slider.on_change('value_throttled', update_data)

    inputs = column(file_input, channel_slider, current_channel_name, row(interval_input, highpass_input, lowpass_input, notch_input, ))

    doc.add_root(column(inputs, p))


bp = Blueprint("spectrum checker", __name__, url_prefix="/spectrum")

@bp.route("/", methods=["GET"])
def bkapp_page():
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    script = server_document("http://localhost:5006/spectrum_bkapp")
    return render_template("spectrum.html", script=script, template="Flask", port=config['PORT'])


# def bk_worker():
#     server = Server({'/spectrum_bkapp': spectrum_bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
#     server.start()
#     server.io_loop.start()
#
#
# Thread(target=bk_worker).start()



