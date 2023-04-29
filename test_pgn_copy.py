from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import *
import pandas as pd
from pathlib import PureWindowsPath, PurePosixPath,Path

N_CLASSES = 20
DATA_DIR ="D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/" #'./datasets/CIHP'
# LIST_PATH = "D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/list/val.txt"#'./datasets/CIHP/list/val.txt'
LIST_PATH = "D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/list/val.csv"
DATA_ID_LIST = "D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/list/val_id.txt"#./datasets/CIHP/list/val_id.txt'
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines()) 
RESTORE_FROM = "D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\checkpoints\CIHP_pgn"#'./checkpoint/CIHP_pgn'

def resize_images(image_batch,h_orig,h1,w_orig,w1):
    return tf.image.resize(image_batch, tf.stack([tf.cast(tf.multiply(h_orig, h1),tf.int32), tf.cast(tf.multiply(w_orig, w1),tf.int32)]))


def main():
    """Create the model and start the evaluation process."""
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, None, False, False, False, coord)
        image, label, edge_gt = reader.image, reader.label, reader.edge
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    # print(image, label, edge_gt, image_rev)
    # print(image_list)

    image_batch = tf.stack([image, image_rev])
    label_batch = tf.expand_dims(label, axis=0) # Add one batch dimension.
    edge_gt_batch = tf.expand_dims(edge_gt, axis=0)

    # print(image_batch, label_batch, edge_gt_batch)

    h_orig, w_orig = tf.cast(tf.shape(image_batch)[1], tf.float32), tf.cast(tf.shape(image_batch)[2],tf.float32)
    # print(h_orig, w_orig)
    image_batch050 =  resize_images(image_batch,h_orig,0.50,w_orig,0.50)
    image_batch075 =  resize_images(image_batch,h_orig,0.75,w_orig,0.75)
    image_batch125 =  resize_images(image_batch,h_orig,1.25,w_orig,1.25)
    image_batch150 =  resize_images(image_batch,h_orig,1.50,w_orig,1.50)
    image_batch175 =  resize_images(image_batch,h_orig,1.75,w_orig,1.75)
    # print(image_batch050, image_batch075, image_batch125 ,image_batch150, image_batch175 )
   
    
    # Create network.
    with tf.compat.v1.variable_scope('', reuse=False):
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
    with tf.compat.v1.variable_scope('', reuse=True):
        net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)


    # parsing net
    parsing_out1_050 = net_050.layers['parsing_fc']
    parsing_out1_075 = net_075.layers['parsing_fc']
    parsing_out1_100 = net_100.layers['parsing_fc']
    parsing_out1_125 = net_125.layers['parsing_fc']
    parsing_out1_150 = net_150.layers['parsing_fc']
    parsing_out1_175 = net_175.layers['parsing_fc']

    parsing_out2_050 = net_050.layers['parsing_rf_fc']
    parsing_out2_075 = net_075.layers['parsing_rf_fc']
    parsing_out2_100 = net_100.layers['parsing_rf_fc']
    parsing_out2_125 = net_125.layers['parsing_rf_fc']
    parsing_out2_150 = net_150.layers['parsing_rf_fc']
    parsing_out2_175 = net_175.layers['parsing_rf_fc']

    # edge net
    edge_out2_100 = net_100.layers['edge_rf_fc']
    edge_out2_125 = net_125.layers['edge_rf_fc']
    edge_out2_150 = net_150.layers['edge_rf_fc']
    edge_out2_175 = net_175.layers['edge_rf_fc']


    # print("#####tf.shape(image_batch)[1:3,]:   ", tf.shape(image_batch)[1:3,])
    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                            tf.image.resize(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


    edge_out2_100 = tf.image.resize(edge_out2_100, tf.shape(image_batch)[1:3,])
    edge_out2_125 = tf.image.resize(edge_out2_125, tf.shape(image_batch)[1:3,])
    edge_out2_150 = tf.image.resize(edge_out2_150, tf.shape(image_batch)[1:3,])
    edge_out2_175 = tf.image.resize(edge_out2_175, tf.shape(image_batch)[1:3,])
    edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)
                                           
    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, axis=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, axis=3) # Create 4-d tensor.


    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, axis=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # prepare ground truth 
    preds = tf.reshape(pred_all, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, N_CLASSES - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_op_iou = tf.compat.v1.metrics.mean_iou(preds, gt, num_classes=N_CLASSES, weights=weights)
    macc, update_op_acc = tf.compat.v1.metrics.accuracy(preds, gt, weights=weights)

    # precision and recall
    recall, update_op_recall = tf.compat.v1.metrics.recall(res_edge, edge_gt_batch)
    # precision, update_op_precision = tf.contrib.metrics.streaming_precision(res_edge, edge_gt_batch)
    precision, update_op_precision = tf.compat.v1.metrics.precision(res_edge, edge_gt_batch)

    update_op = tf.group(update_op_iou, update_op_acc, update_op_recall, update_op_precision)


    # Which variables to load.
    restore_var = tf.compat.v1.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.compat.v1.local_variables_initializer())
    
    # Load weights.

    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # evaluate prosessing
    #../output/cihp_parsing_maps/MEN

    # val='out_path'
    # out_path = f'D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\datasets\CIHP\list\{val}.txt'
    # out_path=str(PurePosixPath(PureWindowsPath(out_path)))
    # out_path =out_path.replace('D:\/', 'D:/')

    df = pd.read_csv(LIST_PATH)
    parsing_dir = list( df['out_root'])

    img_id_list = list( df['f_name'])
    NUM_STEPS = len(img_id_list)

    # out_list = []
    # with open(val_path, 'r') as f:
    #     out_list = f.readlines()

    # parsing_dir =out_list[10].strip()

    # if not os.path.exists(parsing_dir):
    #     os.makedirs(parsing_dir)

    # parsing_dir = './output/cihp_parsing_maps'
    # if not os.path.exists(parsing_dir):
    #     os.makedirs(parsing_dir)

    edge_dir = './output/cihp_edge_maps'
    if not os.path.exists(edge_dir):
        os.makedirs(edge_dir)



    print('____________________________________________________________________________________')
    # print(pred_all.shape)
    # Iterate over training steps.
    for step in range(NUM_STEPS): #NUM_STEPS
        print(step)


        if not os.path.exists(parsing_dir[step]):
            os.makedirs(parsing_dir[step])

        parsing_, scores, edge_, _ = sess.run([pred_all, pred_scores, pred_edge, update_op])

        # if step % 1 == 0:
        #     print('step {:d}'.format(step))
        #     print (image_list[step])
        # img_split = image_list[step].split('/')
        # img_id = img_split[-1][:-4]
        
        msk = decode_labels(parsing_, num_classes=N_CLASSES)

        parsing_im = Image.fromarray(msk[0])

        parsing_im.save('{}/{}.png'.format(parsing_dir[step], img_id_list[step]))

        # print("here")
        # parsing_im.save('{}/{}.png'.format(parsing_dir, img_id))
        # cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])
        # sio.savemat('{}/{}.mat'.format(parsing_dir, img_id), {'data': scores[0,:,:]})
        
        # cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0,:,:,0] * 255)

        print("here")

    # res_mIou = mIoU.eval(session=sess)
    # res_macc = macc.eval(session=sess)
    # res_recall = recall.eval(session=sess)
    # res_precision = precision.eval(session=sess)
    # f1 = 2 * res_precision * res_recall / (res_precision + res_recall)
    # print('Mean IoU: {:.4f}, Mean Acc: {:.4f}'.format(res_mIou, res_macc))
    # print('Recall: {:.4f}, Precision: {:.4f}, F1 score: {:.4f}'.format(res_recall, res_precision, f1))

    coord.request_stop()
    coord.join(threads)
    


if __name__ == '__main__':
    main()


##############################################################333
