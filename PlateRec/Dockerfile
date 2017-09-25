FROM dexxpl33988012.xxp.sap.corp:11000/icn-ml/tf12-py2-3-cuda8

MAINTAINER Andy Zhang <andy.zhang02@sap.com>

ADD TFRecords /nfs/users/matthew/workdir/
ADD tool /nfs/users/matthew/workdir/
ADD binary_classification_CNN.py /nfs/users/matthew/workdir/

WORKDIR "/nfs/users/matthew/workdir"
CMD ["python3", "binary_classification_CNN.py"]