from flask import Flask
from blueprints import index_bp, signal_checker_bp, spectrum_checker_bp, cfc_checker_bp, \
    signal_bkapp, spectrum_bkapp, cfc_bkapp
from bokeh.server.server import Server
from threading import Thread
from tornado.ioloop import IOLoop
import yaml
import argparse

app = Flask(__name__)

app.register_blueprint(index_bp)
app.register_blueprint(signal_checker_bp)
app.register_blueprint(spectrum_checker_bp)
app.register_blueprint(cfc_checker_bp)

with open("./config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)
    config_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8000, type=int,
                        help='Select port for you server to run on')
    parser.add_argument('-f', '--file', default='none', type=str,
                        help='Select the base file for your server')

    args = parser.parse_args()

    config['PORT'] = args.port
    config['FILENAME'] = args.file if args.file.lower() != 'none' else config['FILENAME']

    with open("./config.yaml", 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)
        config_file.close()


    def bk_worker():
        server = Server({'/signal_bkapp': signal_bkapp,
                         '/spectrum_bkapp': spectrum_bkapp,
                         '/cfc_bkapp': cfc_bkapp},
                        io_loop=IOLoop(), allow_websocket_origin=["localhost:"+str(config['PORT'])])
        server.start()
        server.io_loop.start()


    Thread(target=bk_worker).start()

    app.run(port=config['PORT'])