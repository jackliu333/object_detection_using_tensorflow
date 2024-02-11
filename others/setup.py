import os

os.system("pip install protobuf-compiler")
# os.system("cd models/research")
os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=.")
os.system("cd Tensorflow/models/research && cp object_detection/packages/tf2/setup.py .")
os.system("cd Tensorflow/models/research && python -m pip install .")

