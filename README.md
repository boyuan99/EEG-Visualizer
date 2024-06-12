# EEG-Visualizer

<p align="center">
  <a href="http://docs.bokeh.org/en/latest/"><img alt="Bokeh" src="https://img.shields.io/badge/host-bokeh_server-orange.svg?style=flat-square&logo=bokeh"></a>
  <a href="https://flask.palletsprojects.com/en/2.2.x/"><img alt="Flask" src="https://img.shields.io/badge/embed-flask-red.svg?style=flat-square&logo=flask"></a>
  <a href="https://boyuan.io/research/EEG_Visualizer/html/intro.html"><img alt="Online" src="https://img.shields.io/badge/book-EEG_Visualizer-green.svg?style=flat-square"></a></p>


This is a EEG (especially SEEG) signal visualizer tool based on [Flask](https://flask.palletsprojects.com/en/2.2.x/) & [Bokeh Server](https://docs.bokeh.org/en/latest/docs/user_guide/server.html).

SEEG signals often have **multiple channels**, **high sampling frequencies**, and **extended recording periods**, resulting in a large computational drain on the computer. In this visualization tool, we can focus on the exact part of the signal you are observing, so that computational resources can only process that part, thus shortening the loading and processing process.

## Setting Up & Run

The following command will install the packages according to the configuration file `requirements.txt`.

```bash
pip install -r requirements.txt
```

To use the server, run:

```bash
python app.py
```

For more Instructions, please go to [Intro to EEG Visualizer](https://boyuan.io/research/EEG_Visualizer/html/index.html).

## Demo data

Demo data in the **EEG Visualizer** comes from the SEEG data for the results in "[BrainQuake](https://github.com/HongLabTHU/BrainQuake): an open-source Python toolbox for Stereo-EEG analysis“.

https://zenodo.org/record/5675459#.YySZ3uzMK3Y

