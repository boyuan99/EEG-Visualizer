from flask import render_template, Blueprint

import mne
from scipy import signal
from bokeh.embed import server_document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, HoverTool
from bokeh.plotting import figure


def spectrum_bkapp(doc):
    file_base_47 = 'E:/Subject_4_event_7_base.edf'
    raw_base_47 = mne.io.read_raw_edf(file_base_47)
    data_base_47 = raw_base_47.get_data()[:48]

    file_pain_47 = 'E:/Subject_4_event_7_pain.edf'
    raw_pain_47 = mne.io.read_raw_edf(file_pain_47)
    data_pain_47 = raw_pain_47.get_data()[:48]

    file_base_42 = 'E:/Subject_4_event_2_base.edf'
    raw_base_42 = mne.io.read_raw_edf(file_base_42)
    data_base_42 = raw_base_42.get_data()[:48]


    file_pain_42 = 'E:/Subject_4_event_2_pain.edf'
    raw_pain_42 = mne.io.read_raw_edf(file_pain_42)
    data_pain_42 = raw_pain_42.get_data()[:48]

    file_base_46 = 'E:/Subject_4_event_6_base.edf'
    raw_base_46 = mne.io.read_raw_edf(file_base_46)
    data_base_46 = raw_base_46.get_data()[:48]

    file_pain_46 = 'E:/Subject_4_event_6_pain.edf'
    raw_pain_46 = mne.io.read_raw_edf(file_pain_46)
    data_pain_46 = raw_pain_46.get_data()[:48]

    file_base_48 = 'E:/Subject_4_event_8_base.edf'
    raw_base_48 = mne.io.read_raw_edf(file_base_48)
    data_base_48 = raw_base_48.get_data()[:48]

    file_pain_48 = 'E:/Subject_4_event_8_pain.edf'
    raw_pain_48 = mne.io.read_raw_edf(file_pain_48)
    data_pain_48 = raw_pain_48.get_data()[:48]

    raw_base = raw_base_47
    raw_pain = raw_pain_47
    data_base = data_base_47
    data_pain = data_pain_47

    fs = raw_base.info['sfreq']

    b_notch, a_notch = signal.iirnotch(60, 30, fs)
    y_notched_base = signal.filtfilt(b_notch, a_notch, data_base[0])
    y_notched_pain = signal.filtfilt(b_notch, a_notch, data_pain[0])
    f_base, den_base = signal.welch(y_notched_base, fs, nperseg=10240)
    f_pain, den_pain = signal.welch(y_notched_pain, fs, nperseg=10240)

    source = ColumnDataSource(
        data=dict(
            x_base=f_base,
            y_base=den_base,
            x_pain=f_pain,
            y_pain=den_pain
        )
    )
    hover = HoverTool(
        tooltips=[
            ("(x,y)", "($x, $y)"),
        ]
    )
    p = figure(height=400, width=900)
    p.add_tools(hover)
    p.line('x_base', 'y_base', source=source, line_color='skyblue', legend_label='base')
    p.line('x_pain', 'y_pain', source=source, line_color='orange', legend_label='pain')
    p.legend.click_policy = 'hide'


    base_slider = Slider(value=0, start=0, end=47, step=1, width=900, title='Base')
    pain_slider = Slider(value=0, start=0, end=47, step=1, width=900, title='Pain')
    file_selector = Select(title="Sample:", value="Subject_4_Event_7", options=["Subject_4_Event_2",
                                                                                "Subject_4_Event_6",
                                                                                "Subject_4_Event_7",
                                                                                "Subject_4_Event_8"])

    def update_data(attribute, old, new):
        base_channel_update = base_slider.value
        pain_channel_update = pain_slider.value
        file_update = file_selector.value

        if file_update == "Subject_4_Event_2":
            raw_base = raw_base_42
            raw_pain = raw_pain_42
            data_base = data_base_42
            data_pain = data_pain_42
        elif file_update == "Subject_4_Event_6":
            raw_base = raw_base_46
            raw_pain = raw_pain_46
            data_base = data_base_46
            data_pain = data_pain_46
        elif file_update == "Subject_4_Event_7":
            raw_base = raw_base_47
            raw_pain = raw_pain_47
            data_base = data_base_47
            data_pain = data_pain_47
        elif file_update == "Subject_4_Event_8":
            raw_base = raw_base_48
            raw_pain = raw_pain_48
            data_base = data_base_48
            data_pain = data_pain_48

        fs = raw_base.info['sfreq']

        b_notch, a_notch = signal.iirnotch(60, 30, fs)
        y_notched_base = signal.filtfilt(b_notch, a_notch, data_base[base_channel_update])
        y_notched_pain = signal.filtfilt(b_notch, a_notch, data_pain[pain_channel_update])
        f_base, den_base = signal.welch(y_notched_base, fs, nperseg=10240)
        f_pain, den_pain = signal.welch(y_notched_pain, fs, nperseg=10240)

        source.data = dict(
            x_base=f_base,
            y_base=den_base,
            x_pain=f_pain,
            y_pain=den_pain
        )

    for w in [file_selector, base_slider, pain_slider]:
        w.on_change('value', update_data)

    inputs = column(file_selector, base_slider, pain_slider)

    doc.add_root(column(inputs, p))


bp = Blueprint("spectrum checker", __name__, url_prefix="/spectrum")

@bp.route("/", methods=["GET"])
def bkapp_page():
    script = server_document("http://localhost:5006/spectrum_bkapp")
    return render_template("embed.html", script=script, template="Flask ")


# def bk_worker():
#     server = Server({'/spectrum_bkapp': spectrum_bkapp}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:8000"])
#     server.start()
#     server.io_loop.start()
#
#
# Thread(target=bk_worker).start()



