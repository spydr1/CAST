import tensorflow as tf
import numpy as np
from research.slim.nets.mobilenet import mobilenet_v2


from tensorflow.contrib import slim
from nets import resnet_v1


## expanded
import research.slim.nets.mobilenet.conv_blocks as ops
from research.slim.nets.mobilenet import mobilenet as lib
expand_input = ops.expand_input_by_factor
##


import sys

#pvanet
sys.path.append('/home/minjun/Jupyter/ocr/tf-pvanet/')
sys.path.append('/home/minjun/Jupyter/ocr/pylib/src/')
from pvanet import pvanet, pvanet_scope
import util
##



tf.app.flags.DEFINE_string('decoder', 'CRAFT', 'decoder type : original, CRAFT')

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)
    if FLAGS.backbone == 'Resnet':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
            end_points['pool2'] = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1/conv3']
            end_points['pool3'] = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1/conv3']
            end_points['pool4'] = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1/conv3']
            end_points['pool5'] = end_points['resnet_v1_50/block4/unit_3/bottleneck_v1/conv3']
            

    elif FLAGS.backbone == 'Mobilenet':
        with slim.arg_scope(mobilenet_v2.training_scope(weight_decay=weight_decay)):
            logits, endpoints = mobilenet_v2.mobilenet_base(images,is_training=is_training)
        end_points = dict()
        end_points['pool2'] = endpoints['layer_4/output']
        end_points['pool3'] = endpoints['layer_6/output']
        end_points['pool4'] = endpoints['layer_13/output']
        end_points['pool5'] = endpoints['layer_19']
        print(end_points['pool2'],end_points['pool3'],end_points['pool4'],end_points['pool5'])
    elif FLAGS.backbone == 'Pvanet' or 'Pvanet2x':
        print("Backbone is Pvanet")
        expansion = 1 if FLAGS.backbone == 'Pvanet' else 2
        with slim.arg_scope(pvanet_scope(is_training,weight_decay=weight_decay)):
            net, end_points = pvanet(images,expansion=expansion)
            end_points['pool2'] = end_points['conv2']
            end_points['pool3'] = end_points['conv3']
            end_points['pool4'] = end_points['conv4']
            end_points['pool5'] = end_points['conv5']

                    
    if FLAGS.decoder == 'CRAFT':
        print("deep decoder")
        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                x = unpool(end_points['pool5'])
                x = tf.concat([x,end_points['pool4']],axis=-1)
                x = slim.conv2d(x, 256, 1)
                x = slim.conv2d(x, 128, 3)
                print("up_4 shape : ",x.shape)

                x = unpool(x)
                x = tf.concat([x,end_points['pool3']],axis=-1)
                x = slim.conv2d(x, 128, 1)
                x = slim.conv2d(x, 64, 3)
                print("up_3 shape : ",x.shape)

                x = unpool(x)
                x = tf.concat([x,end_points['pool2']],axis=-1)
                x = slim.conv2d(x, 64, 1)
                x = slim.conv2d(x, 32, 3)
                print("up_2 shape : ",x.shape)

                x = slim.conv2d(x, 32, 3)
                x = slim.conv2d(x, 32, 3)
                x = slim.conv2d(x, 16, 3)
                x = slim.conv2d(x, 16, 3)

                F_score = slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(x, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                angle_map = (slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
    elif  FLAGS.decoder == 'Expand':
        expand_kwargs = {
        'expansion_size': expand_input(6),       
        'split_expansion': 1,
        'normalizer_fn': slim.batch_norm,
        'residual': True
        }
        batch_norm_params = {'center': True, 'scale': True,'is_training': is_training}

        with tf.variable_scope('decoder') :
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                                        activation_fn=tf.nn.relu6,
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params=batch_norm_params,
                                        weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([ops.expanded_conv],**expand_kwargs):
                    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],padding='SAME'):
                        x = unpool(end_points['pool5'])
                        x = tf.concat([x,end_points['pool4']],axis=-1)
                        x = ops.expanded_conv(x, 48,depthwise_location='input',expansion_size= expand_input(1,divisible_by=1))
                        print("up_4 : ",x.shape)

                        x = unpool(x)
                        x = tf.concat([x,end_points['pool3']],axis=-1)
                        print("up_3 concat : ",x.shape)
                        x = ops.expanded_conv(x, 24)
                        

                        x = unpool(x)
                        x = tf.concat([x,end_points['pool2']],axis=-1)
                        print("up_3 concat : ",x.shape)
                        x = ops.expanded_conv(x, 16)

                        x = slim.conv2d(x, 16, 3)
                        x = slim.conv2d(x, 16, 3)

                        F_score = slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                        # 4 channel of axis aligned bbox and 1 channel rotation angle
                        geo_map = slim.conv2d(x, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                        angle_map = (slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                        F_geometry = tf.concat([geo_map, angle_map], axis=-1)
                        
    elif  FLAGS.decoder == 'CAST':
        expand_kwargs = {
        'expansion_size': expand_input(6),       
        'split_expansion': 1,
        'normalizer_fn': slim.batch_norm,
        'residual': True
        }
        batch_norm_params = {'center': True, 'scale': True,'is_training': is_training}

        with tf.variable_scope('decoder') :
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                                        activation_fn=tf.nn.relu6,
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params=batch_norm_params,
                                        weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([ops.expanded_conv],**expand_kwargs):
                    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],padding='SAME'):
                        x = unpool(end_points['pool5'])
                        x = tf.concat([x,end_points['pool4']],axis=-1)
                        x = ops.expanded_conv(x, 128,expansion_size= expand_input(1,divisible_by=1))
                        x = ops.expanded_conv(x, 96)

                        print("up_4 shape : ",x.shape)
                        
                        x = unpool(x)
                        x = tf.concat([x,end_points['pool3']],axis=-1)
                        print("up_3 concat : ",x.shape)
                        x = ops.expanded_conv(x, 48)
                        x = slim.conv2d(x, 64, 3)
                        print("up_3 shape : ",x.shape)

                        x = unpool(x)
                        x = tf.concat([x,end_points['pool2']],axis=-1)
                        print("up_2 concat : ",x.shape)                        
                        x = slim.conv2d(x, 32, 3)
                        print("up_2 shape : ",x.shape)

                        x = slim.conv2d(x, 16, 3)
                        x = slim.conv2d(x, 16, 3)

                        F_score = slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                        # 4 channel of axis aligned bbox and 1 channel rotation angle
                        geo_map = slim.conv2d(x, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                        angle_map = (slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                        F_geometry = tf.concat([geo_map, angle_map], axis=-1)
           
    else :
        print("original decoder")
        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                for i in range(4):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [None, 128, 64, 32]
                for i in range(4):
                    if i == 0:
                        h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                    if i <= 2:
                        g[i] = unpool(h[i])
                    else:
                        g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
                # here we use a slightly different way for regression part,
                # we first use a sigmoid to limit the regression range, and also
                # this is do with the angle map
                F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
                              
    return F_score, F_geometry


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
