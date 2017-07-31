from __future__ import print_function
import cPickle
import os
import time
import mxnet as mx
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from module import MutableModule
from rcnn.config import config
from rcnn.io import image
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes,bbox_overlaps
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


class Predictor(object):
    def __init__(self, sym_gen,cfg,data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(sym_gen,cfg,data_names, label_names,is_train=False,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))


def im_proposal(predictor, data_batch, data_names, scale):
    data_dict = dict(zip(data_names, data_batch.data))
    output = predictor.predict(data_batch)

    # drop the batch index
    boxes = output['rois_output'].asnumpy()[:, 1:]
    scores = output['rois_score'].asnumpy()

    # transform to original scale
    boxes = boxes / scale

    return scores, boxes, data_dict


def generate_proposals(predictor, test_data, imdb, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    i = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scale = im_info[0, 2]
        scores, boxes, data_dict = im_proposal(predictor, data_batch, data_names, scale)
        t2 = time.time() - t
        t = time.time()

        # assemble proposals
        dets = np.hstack((boxes, scores))
        original_boxes.append(dets)

        # filter proposals
        keep = np.where(dets[:, 4:] > thresh)[0]
        dets = dets[keep, :]
        imdb_boxes.append(dets)

        if vis:
            vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale)

        print('generating %d/%d' % (i + 1, imdb.num_images),
              'proposal %d' % (dets.shape[0]),
              'data %.4fs net %.4fs' % (t1, t2))
        i += 1

    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.root_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('wrote rpn proposals to {}'.format(rpn_file))
    return imdb_boxes


def im_detect(predictor, data_batch, data_names, scale):
    output = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    if config.TEST.HAS_RPN:
        rois = output['rois_output'].asnumpy()[:, 1:]
    else:
        rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
    im_shape = data_dict['data'].shape

    # save output
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    return scores, pred_boxes, data_dict


def pred_eval(predictor, test_data, roidb, vis=False, thresh=1e-3):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """
    data_root = os.getcwd()
    det_file = os.path.join(data_root,'data/cache/dection.pkl')
    if not os.path.exists(det_file):
        
        assert vis or not test_data.shuffle
        data_names = [k[0] for k in test_data.provide_data]

        nms = py_nms_wrapper(config.TEST.NMS)
        num_samples = len(roidb)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[] for _ in range(num_samples)]

        i = 0
        t = time.time()
        for im_info, data_batch in test_data:
            t1 = time.time() - t
            t = time.time()
    
            scale = im_info[0, 2]
            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scale)
            #print(scores)
    
            t2 = time.time() - t
            t = time.time()
            #onlt class 1 is what we wanted
            
            indexes = np.where(scores[:, 1] > thresh)[0]
           
            cls_scores = scores[indexes, 1, np.newaxis]
            cls_boxes = boxes[indexes, 1 * 4:2 * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            all_boxes[i] = cls_dets[keep,:]
            t3 = time.time() - t
            t = time.time()
            print('testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(i, num_samples, t1, t2, t3))
            i += 1

        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(det_file,'rb') as f:
            all_boxes = cPickle.load(f)
    evalutate_detections(all_boxes,roidb)
    vis_all_detection(all_boxes,roidb)

    #imdb.evaluate_detections(all_boxes)
def evalutate_detections(all_boxes,roidb):
    """evalutate detections.
    :param all_boxes:all boxes predicted by our model
    :param roidb:that store the ground truth info
    :param p:select boxes which probability large than p
    :param thresh:overlap threshold."""
    assert len(all_boxes)==len(roidb)
    pos_count = 0
    for  i in range(len(roidb)):
        ground_truth = roidb[i]['bbox']
        pred_boxes = all_boxes[i]
        if pred_boxes.shape[0] == 0:
            continue
        pred_box_ind = np.argmax(pred_boxes[:,4])
        pred_box = pred_boxes[pred_box_ind,:]
        pred_box = pred_box[np.newaxis,:]
        overlap = bbox_overlaps(pred_box[:,:4].astype(np.float),ground_truth.astype(np.float))
        if overlap[0][0] > 0.9:
            pos_count += 1
            
    acc = float(pos_count) / len(roidb)
    print(acc)
        
    


def vis_all_detection(all_boxes,roidb):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """

    for i in range(len(roidb)):
		# show image
        ax = plt.gca()
        plt.figure()
        file_name = roidb[i]['image']
        I = io.imread(file_name)
        ax.imshow(I)
        # show refer expression
        print("raw:{}".format(roidb[i]['sent']))
        gt_box = roidb[i]['bbox'][0]
        gt_box_plot = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0]+1, gt_box[3]-gt_box[1]+1, fill=False, edgecolor='green', linewidth=3)
        ax.add_patch(gt_box_plot)
        pred_boxes = all_boxes[i]
        if pred_boxes.shape[0] == 0:
            continue
        pred_box_ind = np.argmax(pred_boxes[:,4])
        pred_box = pred_boxes[pred_box_ind,:]
        pred_box_plot = Rectangle((pred_box[0], pred_box[1]), pred_box[2]-gt_box[0]+1, gt_box[3]-gt_box[1]+1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(pred_box_plot)
        plt.show()
    plt.show()
        


def draw_all_detection(im_array, detections, class_names, scale):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, config.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
