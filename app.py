from flask import Flask
from blueprints import index_bp, signal_checker_bp, spectrum_checker_bp, cfc_checker_bp, \
    signal_bkapp, spectrum_bkapp, cfc_bkapp
from bokeh.server.server import Server
from threading import Thread
from tornado.ioloop import IOLoop

app = Flask(__name__)

app.register_blueprint(index_bp)
app.register_blueprint(signal_checker_bp)
app.register_blueprint(spectrum_checker_bp)
app.register_blueprint(cfc_checker_bp)


def bk_worker():
    server = Server({'/signal_bkapp': signal_bkapp,
                     '/spectrum_bkapp': spectrum_bkapp,
                     '/cfc_bkapp': cfc_bkapp},
                    io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
    server.start()
    server.io_loop.start()


Thread(target=bk_worker).start()

if __name__ == "__main__":
    app.run(port=8000)