# coding=utf-8
import os
import sys
import time
from flask import request, send_from_directory
from flask import Flask, request, redirect, url_for
import uuid
import cv2
import matplotlib.image as Image
import tensorflow as tf
# from classify_image import run_inference_on_image
from plateRec import PlateRec

ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
# tf.app.flags.DEFINE_string('model_name', 'my_inception_v4_freeze.pb', '')
# tf.app.flags.DEFINE_string('label_file', 'my_inception_v4_freeze.label', '')
# tf.app.flags.DEFINE_string('upload_folder', '/tmp/', '')
# tf.app.flags.DEFINE_integer('num_top_predictions', 5,
#                             """Display this many predictions.""")
# tf.app.flags.DEFINE_integer('port', '5001',
#                             'server with port,if no port, use deault port 80')
#
# tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = os.path.abspath('./static')

ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


def inference(file_name):
    # try:
    #     predictions, top_k, top_names = run_inference_on_image(file_name, model_file=FLAGS.model_name)
    #     print(predictions)
    # except Exception as ex:
    #     print(ex)
    #     return ""
    img = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img

    plate_rec.main()
    img_path = os.path.abspath('./static/plate.jpg')
    Image.imsave(img_path, plate_rec.img_con_sobel)

    new_url = '/static/%s' % os.path.basename(img_path)
    image_tag = '<img src="%s"></img><p>'
    new_tag = image_tag % new_url
    format_string = ''
    # for node_id, human_name in zip(top_k, top_names):
    #     score = predictions[node_id]
    #     format_string += '%s (score:%.5f)<BR>' % (human_name, score)
    # ret_string = new_tag + format_string + '<BR>'
    for plate_str in plate_rec.plate_string:
        format_string += 'Plate is %s' % plate_str
    ret_string = new_tag + format_string + '<BR>'
    return ret_string


@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
    <!doctype html>
    <title>Plate recognition</title>
    <h1>Feed your plate images here</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file value='Choose image'>
         <input type=submit value='Submit'>
    </form>
    <p>%s</p>
    """ % "<br>"
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            # type_name = 'N/A'
            # print('file saved to %s' % file_path)
            out_html = inference(file_path)

            return result + out_html
    return result


if __name__ == "__main__":
    print('listening on port 50050')
    app.run(host='0.0.0.0', port=50050)
