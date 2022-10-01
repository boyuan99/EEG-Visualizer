from flask import Blueprint, render_template
import yaml

bp = Blueprint("index", __name__, url_prefix="/")

@bp.route("/")
def index():
    with open("./config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_file.close()
    return render_template("index.html", template="Flask", port=config['PORT'])
