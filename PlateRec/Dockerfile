FROM dexxpl33988012.xxp.sap.corp:11000/icn-ml/tf12-py2-3-cuda8

MAINTAINER Andy Zhang <andy.zhang02@sap.com>

ADD TFRecords /root/
ADD tool /root/
ADD binary_classification_CNN.py /root/
ADD net_structure /root/

WORKDIR "/root"
CMD ["python3", "binary_classification_CNN.py"]