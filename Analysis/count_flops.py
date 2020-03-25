from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim
import tensorflow as tf


def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
def count(graph,op_name):
    with tf.Session(graph=graph) as sess:
        total_parameters=0
        for variable in tf.trainable_variables():  
            local_parameters=1
            shape = variable.get_shape()  #getting shape of a variable
            for i in shape:
                local_parameters*=i.value  #mutiplying dimension values
            total_parameters+=local_parameters
        #print("-----params-----" , total_parameters)
        sess.run(tf.global_variables_initializer())
        flops = tf.profiler.profile(sess.graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
        flops_1 = flops.total_float_ops

        # ***** (2) freeze graph *****
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(),op_name )
    with tf.gfile.GFile('tmp_graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    frozen_g = load_pb('./tmp_graph.pb')
    with frozen_g.as_default():
        flops = tf.profiler.profile(frozen_g, options = tf.profiler.ProfileOptionBuilder.float_operation())
        #print('FLOP after freezing', flops.total_float_ops)
        flops_2=flops.total_float_ops
    return total_parameters,flops_1,flops_2