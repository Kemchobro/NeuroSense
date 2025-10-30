import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_mdoel('C:\\Users\\vidha\\Documents\\congressionalapp\\pipeline.ipynb')
tflite_model = converter.convert()

with open('model.tflite','wb') as f:
    f.write(tflite_model)