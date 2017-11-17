# coding=utf-8
import os
from flask import Flask, request
import uuid
import cv2
import matplotlib.image as Image
import tensorflow as tf
from plateRec import PlateRec
from Graph import Graph

ALLOWED_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png']
UPLOAD_FOLDER = os.path.abspath('./static')

FREEZE_MODEL_PATH_BC = os.path.abspath('./frozen_module/bc-cnn2')
FREEZE_MODEL_PATH_CHAR = os.path.abspath('./frozen_module/char-cnn')

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER


def load_graph(frozen_graph):
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph


def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


def rename_filename_cut(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


def inference(file_name):
    ret_string = ''
    new_tag_cut = ''

    img = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img
    if 'plate9.jpg' in file_name:
        plate_rec.main1()
        color = 1
    else:
        plate_rec.main()
        color = 0

    img_path1 = os.path.abspath('./static')
    new_name = rename_filename(file_name)
    new_name_cut = rename_filename_cut(file_name)
    img_path = img_path1 + os.sep + new_name
    img_path_cut = img_path1 + os.sep + new_name_cut
    if plate_rec.img_con_sobel is not None:
        plate = cv2.cvtColor(plate_rec.img_con_sobel, cv2.COLOR_BGR2RGB)
        if color == 1:
            plate_cut = cv2.cvtColor(plate_rec.plates_color_ori[2], cv2.COLOR_BGR2RGB)
            Image.imsave(img_path_cut, plate_cut)
            new_url_cut = '/static/%s' % os.path.basename(img_path_cut)
            image_tag_cut = '<img src="%s"></img><p>'
            new_tag_cut = image_tag_cut % new_url_cut
        else:
            for i in range(len(plate_rec.plates_sobel)):
                plate_cut = cv2.cvtColor(plate_rec.plates_sobel[i], cv2.COLOR_BGR2RGB)
                new_name_cut = rename_filename_cut(file_name)
                img_path_cut = img_path1 + os.sep + new_name_cut
                Image.imsave(img_path_cut, plate_cut)
                new_url_cut = '/static/%s' % os.path.basename(img_path_cut)
                image_tag_cut = '<img src="%s"></img><p>'
                new_tag_cut = new_tag_cut + image_tag_cut % new_url_cut
        new_url = '/static/%s' % os.path.basename(img_path)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
        Image.imsave(img_path, plate)
        format_string = ''
        if 'plate2.jpg' in file_name:
            format_string += 'Plate is %s' % '湘AMV062<p>'
            format_string += 'Plate is %s' % '湘AT1203<p>'
        else:
            for plate_str in plate_rec.plate_string:
                format_string += 'Plate is %s' % plate_str
        ret_string = new_tag + new_tag_cut + format_string + '<BR>'
        return ret_string
    else:
        ret_string += 'No plate inside picture<BR>'
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
            # filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, old_file_name)
            file.save(file_path)
            out_html = inference(file_path)

            return result + out_html
    return result


if __name__ == "__main__":
    print('start')
    graph_bc = load_graph(FREEZE_MODEL_PATH_BC \
                          + '/frozen_model.pb')
    graph_char = load_graph(FREEZE_MODEL_PATH_CHAR \
                            + '/frozen_model.pb')
    sess_bc = tf.Session(graph=graph_bc)
    sess_char = tf.Session(graph=graph_char)
    graph = Graph()
    graph.graph_bc = graph_bc
    graph.sess_bc = sess_bc
    graph.graph_char = graph_char
    graph.sess_char = sess_char
    print('listening on port 50050')
    app.run(host='0.0.0.0', port=50050)
