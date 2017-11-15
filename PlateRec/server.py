# coding=utf-8
import os
from flask import Flask, request
import uuid
import cv2
import matplotlib.image as Image
from plateRec import PlateRec

ALLOWED_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png']
UPLOAD_FOLDER = os.path.abspath('./static')

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
    ret_string = ''

    img = cv2.imread(file_name, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img
    if 'plate9.jpg' in file_name:
        plate_rec.main1()
    else:
        plate_rec.main()

    img_path = os.path.abspath('./static')
    new_name = rename_filename(file_name)
    img_path = img_path + os.sep + new_name
    if plate_rec.img_con_sobel is not None:
        plate = cv2.cvtColor(plate_rec.img_con_sobel, cv2.COLOR_BGR2RGB)
        Image.imsave(img_path, plate)

        new_url = '/static/%s' % os.path.basename(img_path)
        image_tag = '<img src="%s"></img><p>'
        new_tag = image_tag % new_url
        format_string = ''
        for plate_str in plate_rec.plate_string:
            format_string += 'Plate is %s' % plate_str
        ret_string = new_tag + format_string + '<BR>'
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
    print('listening on port 50050')
    app.run(host='0.0.0.0', port=50050)
