from flask import render_template, Blueprint
from threading import Thread
from tornado.ioloop import IOLoop
import numpy as np
import mne
import yaml
from .source_utils import FrequencyAnalysis as fa
from .source_utils import RawDF, SpectrumDF, FrequencyAnalysis
from .bp_utils import stackc_tick_loc, stack_x_axis_times, stack_y_axis_signals, convolve_multichannel
from bokeh.embed import server_document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, RangeSlider, TextInput, MultiChoice
from bokeh.plotting import figure
from bokeh.server.server import Server


def signal_bkapp(doc):
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()

    # import os
    # print(os.listdir('./'))

    raw_edf = RawDF(filename=config['FILENAME'])
    offset = 0.0001
    ch_num = raw_edf.nchan
    p = figure(height=900, width=900, title="sEEG Visualization")
    source = ColumnDataSource(data=dict(x_base=stack_x_axis_times(raw_edf.raw.times[:3000], ch_num),
                                        y_base=stack_y_axis_signals(raw_edf.raw.get_data()[:, :3000], ch_num, offset),
                                        x_pain=stack_x_axis_times(raw_edf.raw.times[:3000], ch_num),
                                        y_pain=stack_y_axis_signals(raw_edf.raw.get_data()[:, :3000], ch_num, offset),))
    p.multi_line('x_base', 'y_base', source=source, line_color='skyblue', legend_label='base')
    p.multi_line('x_pain', 'y_pain', source=source, line_color='orange', legend_label='pain')
    y_tick_loc = stackc_tick_loc(raw_edf.raw.get_data(), ch_num, offset)
    y_tick_labels = raw_edf.ch_names
    y_tick_dict = dict(zip(y_tick_loc, y_tick_labels))
    p.yaxis.ticker = y_tick_loc
    p.yaxis.major_label_overrides = y_tick_dict
    p.legend.click_policy = "hide"

    range_slider = RangeSlider(start=0, end=10000, value=(0, 3000), step=500, width=900, title="Range Slider")
    offset_slider = Slider(value=0.0001, start=0, end=0.002, step=0.0001, width=900, title='Offset')
    start_slider = Slider(value=0, start=0, end=raw_edf.raw.get_data().shape[1] - 10000, step=5000, width=900, title='Start From')
    smooth_slider = Slider(value=0, start=0, end=1000, step=10, width=900, title='Smooth Window')
    multi_choice = MultiChoice(value=list(raw_edf.brain_regions), options=list(raw_edf.brain_regions), title='Brain Regions')
    highpass_input = TextInput(title='Highpass Filter:', value='None')
    lowpass_input = TextInput(title='Lowpass Filter:', value='None')
    bandpass_input = TextInput(title='Bandpass Filter:', value='None')
    notch_input = TextInput(title='Notch Filter:', value='None')

    file_input = TextInput(title='Compare File:', value='Default')

    def update_data(attribute, old, new):
        range_update = range_slider.value
        offset_update = offset_slider.value
        file_update = file_input.value
        start_update = start_slider.value
        kernel_size_update = smooth_slider.value
        multi_choice_update = multi_choice.value
        highpass_update = highpass_input.value
        lowpass_update = lowpass_input.value
        bandpass_update = bandpass_input.value
        notch_update = notch_input.value

        if kernel_size_update == 0:
            kernel = [-1, -1]
        else:
            kernel = np.ones(kernel_size_update) / kernel_size_update

        if file_update.lower() == "default":
            raw_base = raw_edf.raw
            chan_base, data_base = raw_edf.filter_data_from_region(multi_choice_update)
            raw_noba = raw_edf.raw
            chan_noba, data_noba = raw_edf.filter_data_from_region(multi_choice_update)

        ch_num = len(chan_base)
        y_base_filted = convolve_multichannel(
            data_base[0:ch_num, start_update + range_update[0]:start_update + range_update[1]],
            kernel, 0)
        y_noba_filted = convolve_multichannel(
            data_noba[0:ch_num, start_update + range_update[0]:start_update + range_update[1]],
            kernel, 0)


        if highpass_update != 'None':
            try:
                cutoff = float(highpass_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.butter_highpass_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.butter_highpass_filter(y_noba_filted, cutoff, raw_edf.freq)
            print(y_noba_filted[0][0])

        else:
            pass


        if lowpass_update != 'None':
            try:
                cutoff = float(lowpass_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.butter_lowpass_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.butter_lowpass_filter(y_noba_filted, cutoff, raw_edf.freq)
            print(y_noba_filted[0][0])
        else:
            pass


        if notch_update != 'None':
            try:
                cutoff = float(notch_update)
            except ValueError:
                print("Could not convert this input to float!")

            y_base_filted = fa.notch_filter(y_base_filted, cutoff, raw_edf.freq)
            y_noba_filted = fa.notch_filter(y_noba_filted, cutoff, raw_edf.freq)
            print(y_noba_filted[0][0])
        else:
            pass


        source.data = dict(
            x_base=stack_x_axis_times(raw_base.times[start_update + range_update[0]:start_update + range_update[1]],
                                      ch_num),
            y_base=stack_y_axis_signals(y_base_filted, ch_num, offset_update),
            x_pain=stack_x_axis_times(raw_noba.times[start_update + range_update[0]:start_update + range_update[1]],
                                      ch_num),
            y_pain=stack_y_axis_signals(y_noba_filted, ch_num, offset_update))

        y_tick_loc = stackc_tick_loc(data_base, ch_num, offset)
        y_tick_labels = chan_base
        y_tick_dict = dict(zip(y_tick_loc, y_tick_labels))
        p.yaxis.ticker = y_tick_loc
        p.yaxis.major_label_overrides = y_tick_dict

    for w in [file_input, multi_choice, highpass_input, lowpass_input, notch_input]:
        w.on_change('value', update_data)
    for w in [range_slider, offset_slider, smooth_slider, start_slider]:
        w.on_change('value_throttled', update_data)

    inputs = column(file_input, range_slider, start_slider, smooth_slider, offset_slider, multi_choice,
                    row(highpass_input, lowpass_input, notch_input))

    doc.add_root(row(inputs, p))

bp = Blueprint("signal checker", __name__, url_prefix='/signal')

@bp.route("/", methods=['GET'])
def bkapp_page():
    script = server_document("http://localhost:5006/signal_bkapp")
    return render_template("embed.html", script=script, template="Flask")

# if __name__ == "__main__":
#     def bk_worker():
#         server = Server({'/signal_bkapp': signal_bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
#         server.start()
#         server.io_loop.start()
#
#
#     Thread(target=bk_worker).start()
