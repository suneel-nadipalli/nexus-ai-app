from flask import Flask, request, render_template, jsonify

from glob import glob

from pathlib import Path

from scripts import mongo_utils

from scripts import rag_utils

from flask_cors import CORS

from dotenv import load_dotenv

import os

load_dotenv()

app = Flask(__name__)

CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

client = mongo_utils.connect_to_mongo()

print("Connected to MongoDB")

def captitalize_name(name):
    name_split = name.split("_")

    return " ".join([x.capitalize() for x in name_split])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/story-options')
def new_page():

    image_paths = glob("static/images/stories/*")

    names = [Path(image).stem for image in image_paths]

    labels = [captitalize_name(name) for name in names]

    image_label_pairs = zip(image_paths, labels)

    return render_template('story_options.html', image_label_pairs=image_label_pairs)

@app.route('/story-sel')
def destination_page():
    label = request.args.get('label')

    name = Path(label).stem

    pages = mongo_utils.get_pages(name, client)

    page_txts = [page["text"] for page in pages]

    return render_template('story_sel.html', pages=page_txts, pdf_path=name, title=captitalize_name(name), bkg_path=name)


@app.route('/summ', methods=['POST'])
def summarize():
    pdf_path = request.json['pdf_path']

    text = request.json['text']

    vs = mongo_utils.get_vs(pdf_path, client)

    summary = rag_utils.summ(vs, text)

    return jsonify({'summary': summary})

@app.route('/clf', methods=['POST'])
def classify():

    pdf_path = request.json['pdf_path']

    text = request.json['text']

    vs = mongo_utils.get_vs(pdf_path, client)
    
    text = request.json['text']

    decision = rag_utils.clf_seq(vs, text).lower()

    return jsonify({'decision': decision})

@app.route('/options', methods=['POST'])    
def options():
    pdf_path = request.json['pdf_path']

    text = request.json['text']

    vs = mongo_utils.get_vs(pdf_path, client)

    options = eval(rag_utils.gen_options(vs, text))

    return jsonify({'options': options})

@app.route('/path', methods=['POST'])
def path():
    pdf_path = request.json['pdf_path']

    text = request.json['text']

    decision = request.json['decision']

    vs = mongo_utils.get_vs(pdf_path, client)

    path = rag_utils.gen_path(vs, text, decision)

    return jsonify({'path': path})


if __name__ == '__main__':
    app.run()