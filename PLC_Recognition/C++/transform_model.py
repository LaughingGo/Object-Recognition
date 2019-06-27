from keras import backend as K
from tensorflow.python.framework import graph_util
import tensorflow as tf

def export_to_pb(model,model_path):
    # Here model is a Mask R-CNN model

    # model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/logs/plc20180802T2201/mask_rcnn_plc_0150.h5'
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # All new operations will be in test mode from now on.
   
    K.set_learning_phase(0)

    # The filename is built from concantenating the decoders
    filename = os.path.splitext(os.path.basename(model_path))[0] + ".pb"

    # Get the TF session
    sess = K.get_session()

    # Get keras model and save
    model_keras = model.keras_model

    # Get the output heads name
    output_names_all = [output.name.split(':')[0] for output in model_keras.outputs]

    # Getthe graph to export
    graph_to_export = sess.graph

    # Freeze the variables in the graph and remove heads that were not selected
    # this will also cause the pb file to contain all the constant weights
    od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                             graph_to_export.as_graph_def(),
                                                             output_names_all)

    model_dirpath = os.path.dirname(model_path)
    pb_filepath = os.path.join(model_dirpath, filename)
    print('Saving frozen graph {} ...'.format(os.path.basename(pb_filepath)))

    frozen_graph_path = pb_filepath
    with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(od_graph_def.SerializeToString())
    print('{} ops in the frozen graph.'.format(len(od_graph_def.node)))
    print()

