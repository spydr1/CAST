import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import locality_aware_nms as nms_locality
import lanms
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import zipfile
from PIL import Image

import sys 
sys.path.append('/home/minjun/Jupyter/ocr/EAST/Analysis/')
import count_flops 
import random
import colorsys

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('gpu_use', True, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', None, '')
tf.app.flags.DEFINE_string('output_dir', '.', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
#tf.app.flags.DEFINE_string('dataset', 'icdar15', '')
tf.app.flags.DEFINE_string('backbone', 'Mobilenet', 'what kind of backbone')
tf.app.flags.DEFINE_integer('weight_list', 1, '')
tf.app.flags.DEFINE_integer('max_side_len', None, '')

tf.app.flags.DEFINE_float('score_map_thresh', 0.8, '')
tf.app.flags.DEFINE_float('box_thresh', 0.1, '')
tf.app.flags.DEFINE_float('nms_thres', 0.2, '')
tf.app.flags.DEFINE_bool('summary', False, '')

import model
from icdar import restore_rectangle
import re

FLAGS = tf.app.flags.FLAGS

score_map_thresh=FLAGS.score_map_thresh
box_thresh=FLAGS.box_thresh
nms_thres=FLAGS.nms_thres
def random_colors(N, bright=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def get_images(test_data_path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG','gif']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


if FLAGS.dataset == 'icdar15':
    FLAGS.max_side_len = 1280
    print("max size : ",FLAGS.max_side_len)
elif FLAGS.dataset == 'icdar17_mlt':
    FLAGS.max_side_len = 2400
    print("max size : ",FLAGS.max_side_len)
elif FLAGS.dataset == 'icdar13':
    FLAGS.max_side_len = 256
    print("max size : ",FLAGS.max_side_len)


def resize_image(im, max_side_len=FLAGS.max_side_len):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = max_side_len/max(resize_h, resize_w)
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)



def detect(score_map, geo_map, timer, score_map_thresh=score_map_thresh, box_thresh=box_thresh, nms_thres=nms_thres):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    print ("dataset : ",FLAGS.dataset)


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

            
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        if FLAGS.summary == True : 
            img_dir='/home/minjun/Jupyter/data/ocr/2017_MLT/test/ts_img_00001.jpg'
            im = cv2.imread(img_dir)[:, :, ::-1]
            im_resized, (ratio_h, ratio_w) = resize_image(im)

        


        with tf.Session() as sess:
            #flops = tf.profiler.profile(sess.graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
            #print('FLOP before freezing', flops.total_float_ops)
            for w in range(1,FLAGS.weight_list+1):
             
                if FLAGS.dataset == 'icdar15':
                    test_data_path = '/home/minjun/Jupyter/data/ocr/2015/test/'
                    basedir = FLAGS.output_dir+'/output/tmp15/'
                    resfolder = FLAGS.output_dir+'/output/res15/'
                    imgfolder = FLAGS.output_dir+'/output/res_img_15/'

                elif FLAGS.dataset == 'icdar13':
                    print('icdar13')
                    test_data_path = '/home/minjun/Jupyter/data/ocr/2013/Text_Localization/test/'
                    basedir = FLAGS.output_dir+'/output/tmp13/'
                    resfolder = FLAGS.output_dir+'/output/res13/'
                    imgfolder = FLAGS.output_dir+'/output/res_img_13/'
                elif FLAGS.dataset == 'icdar17_mlt':
                    print('icdar17_mlt')
                    test_data_path = '/home/minjun/Jupyter/data/ocr/2017_MLT/test/'
                    basedir = FLAGS.output_dir+'/output/tmp17_mlt/'
                    resfolder = FLAGS.output_dir+'/output/res17_mlt/'
                    imgfolder = FLAGS.output_dir+'/output/res_img_17_mlt/'
                else : 
                    test_data_path = FLAGS.test_data_path
                    basedir = './output/tmp/'
                    resfolder = './output/res/'
                    imgfolder = './output/img/'
                    
                if not os.path.isdir(basedir):
                    os.mkdir(basedir)
                    os.mkdir(resfolder)
                    os.mkdir(imgfolder)
                    
                if os.listdir(resfolder) is not None :
                    resname =[int(name.split('.')[0]) for name in os.listdir(resfolder) if '.zip' in name]
                    resname.sort()
                    if len(resname)>0:
                        print("last file :", resname[-1])
                        resname =resname[-1]+1
                    else : resname = 1
                else : 
                    resname=1
                        
                if FLAGS.checkpoint_path is not None :
                    ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                    #model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                    model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.all_model_checkpoint_paths[-w]))
                    print('Restore from {}'.format(model_path))
                    saver.restore(sess, model_path)
                else: 
                    sess.run(tf.global_variables_initializer())
                    print('checkpoint_path is None')

                im_fn_list = get_images(test_data_path)
                t = time.time()
                net_t = 0
                for k,im_fn in enumerate(im_fn_list):
                    print("Test image {:d}/{:d}".format(k+1, len(im_fn_list)), end='\r')
                    if im_fn.endswith('gif') : 
                        gif = cv2.VideoCapture(im_fn)
                        ret,frame = gif.read() 
                        im = Image.fromarray(frame)
                        im = im.convert('RGB')
                        pixels = list(im.getdata())
                        width, height = im.size
                        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
                        im = np.array(pixels, dtype=np.uint8)
                    else : 
                        im = cv2.imread(im_fn)[:, :, ::-1]
                    
                        
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im)
                    #print("image size : ",np.shape(im_resized))

                    timer = {'net': 0, 'restore': 0, 'nms': 0,'post_processing':0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    #print(np.shape(score),np.shape(geometry))
                    timer['net'] = time.time() - start
                    net_t += timer['net']

                    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)



                    if boxes is not None:
                        score = boxes[:,8]
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                    duration = time.time() - start_time

                    # save to file
                    timer['post_processing'] = time.time()
                    s = os.path.basename(im_fn).split('.')[0]
                    s = re.findall('\d+', s)[0]
                    if FLAGS.dataset == 'icdar17_mlt' or FLAGS.dataset =='icdar15' or FLAGS.dataset == 'icdar13' :
                        res_file = os.path.join(basedir,'res_img_{}.txt'.format(s))

                    else :
                        res_file = os.path.join(basedir,'res_{}.txt'.format(s))
                    
                    with open(res_file, 'w') as f:
                        if boxes is not None:
                            colors = random_colors(len(boxes))
                            if FLAGS.dataset == 'icdar13':
                                for i,box in enumerate(boxes):
                                    #color = np.random.randint(256, size=3)
                                    color = colors[i]
                                    box = sort_poly(box.astype(np.int32))
                                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                        continue
                                    y_max = max(box[0, 1], box[1, 1], box[2, 1], box[3, 1])
                                    y_min = min(box[0, 1], box[1, 1], box[2, 1], box[3, 1])
                                    x_max = max(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
                                    x_min = min(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
                                    f.write('{},{},{},{} \r\n'.format(
                                        x_min,y_min,x_max,y_max
                                    ))
                                    if not FLAGS.no_write_images:
                                        cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color, thickness=2)
                                    if boxes is None : 
                                        print(res_file, "box is none" )
                            elif FLAGS.dataset == 'icdar15': 
                                for i,box in enumerate(boxes):
                                    color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                                    color = (0, 255, 255)
                                    # to avoid submitting errors
                                    #color = colors[i]
                                    #color = np.random.rand(3)
                                    box = sort_poly(box.astype(np.int32))
                                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                        continue
                                    f.write('{},{},{},{},{},{},{},{} \r\n'.format(
                                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]
                                    ))
                                    if not FLAGS.no_write_images:
                                        try : cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color, thickness=2)
                                        except : print("error is raised ",res_file )
                                if boxes is None : 
                                    print(res_file, "box is none" )
                            else : 
                                for i,box in enumerate(boxes):
                                    # to avoid submitting errors
                                    color = np.random.rand(3)
                                    #color = colors[i]
                                    box = sort_poly(box.astype(np.int32))
                                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                        continue
                                    f.write('{},{},{},{},{},{},{},{}, {} \r\n'.format(
                                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],score[i]
                                    ))
                                    if not FLAGS.no_write_images:
                                        try : cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=color, thickness=2)
                                        except : print("error is raised ",res_file )
                                if boxes is None : 
                                    print(res_file, "box is none" )
                            timer['post_processing'] = time.time()-timer['post_processing']
                            #print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms post processing : {:.0f}ms'.format(
                            #im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000,timer['post_processing']*1000))


                        fantasy_zip = zipfile.ZipFile(os.path.join(resfolder,str(resname))+'.zip', 'w')               
                        if not FLAGS.no_write_images:
                            img_path = os.path.join(imgfolder, os.path.basename(im_fn))
                            cv2.imwrite(img_path, im[:, :, ::-1])
                cur_t=time.time()        
                print('time: ' , cur_t - t, 'net time : ', net_t)
                for folder, subfolders, files in os.walk(basedir):
                    for file in files:
                        if file.endswith('.txt'):
                            fantasy_zip.write(os.path.join(folder, file), 
                                              os.path.relpath(os.path.join(folder, file),basedir), 
                                              compress_type = zipfile.ZIP_DEFLATED)
                print('--------------{}----------------'.format(os.path.join(resfolder,str(resname))+'.zip'))
                fantasy_zip.close()


if __name__ == '__main__':
    if FLAGS.gpu_use == False:
        print("CPU Evaluation")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")
    
    tf.app.run()
