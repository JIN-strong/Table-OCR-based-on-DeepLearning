"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2021-09-23
##################################################################################################
"""
import copy
import json
import sys

import os

import pandas as pd

sys.path.append('./model/RARE/')
sys.path.append('./model/LGPMA/')

import numpy as np
from davarocr.davar_common.apis import inference_model, init_model
from ppstructure.table.matcher import distance, compute_iou
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
from ppstructure.utility import parse_args
import cv2
from ppstructure.table.tablepyxl import tablepyxl


def lgpma_ocr(index_list,args_dic):
    # savepath = args['savepath']  # path to save prediction
    checkpoint_file = args_dic['checkpoint_file']  # model path
    config_file = args_dic['config_file']
    excel_path_all = args_dic['output']
    img_path = args_dic['image_dir']
    args = args_gen(args_dic)
    model = init_model(config_file, checkpoint_file)

    image_file_list = get_image_file_list(img_path)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    img_num = len(image_file_list)
    for i, img_path in enumerate(image_file_list):
        result = inference_model(model, img_path)[0]
        img = cv2.imread(img_path)
        dt_boxes, rec_res = dt_boxes_and_rec_res_gen(args, img)
        index=index_list[i]
        pred_html, pred = rebuild_table(result, dt_boxes, rec_res,index)  # main
        # print(type(pred_html))
        excel_path = os.path.join(excel_path_all, os.path.basename(img_path).split('.')[0] + '.xlsx')
        to_excel(pred_html, excel_path)
        # with open("output/1.html", "w", encoding="utf-8") as writer:
        #     json.dump(pred_html, writer, ensure_ascii=False)
        # with open('output/1.html', 'r') as f:
        #     df = pd.read_html('output/1.html', encoding='utf-8')
        #     bb = pd.ExcelWriter('out.xlsx')
        #     df[0].to_excel(bb)
        #     bb.close()

    # pred_dict[sample['filename']] = result['html']


# pred_html, pred = rebuild_table(structure_res, dt_boxes, rec_res) # main
# to_excel(pred_html, excel_path)

def args_gen(arg_dic):
    args = parse_args()
    args.image_dir = arg_dic["image_dir"]
    args.det_model_dir = arg_dic["det_model_dir"]
    args.rec_model_dir = arg_dic["rec_model_dir"]
    args.rec_char_dict_path = arg_dic["rec_char_dict_path"]
    return args


def rebuild_table(structure_res, dt_boxes, rec_res,index):
    pred_structures, pred_bboxes = structure_res['html'], structure_res[
        'bboxes']  # pred_structures, pred_bboxes分别是表结构html和表单元格坐标
    pred_structures = html_change(pred_structures)
    # matched_index = match_result(dt_boxes, pred_bboxes)  # 匹配文字检测框和单元格框
    # print("index",matched_index)
    pred_html, pred = get_pred_html(pred_structures, index, rec_res)

    return pred_html, pred


def dt_boxes_and_rec_res_gen(args, img):
    ori_im = copy.deepcopy(img)
    text_detector = predict_det.TextDetector(args)
    text_recognizer = predict_rec.TextRecognizer(args)
    dt_boxes, elapse = text_detector(copy.deepcopy(img))
    dt_boxes = sorted_boxes(dt_boxes)

    r_boxes = []
    for box in dt_boxes:
        x_min = box[:, 0].min() - 1
        x_max = box[:, 0].max() + 1
        y_min = box[:, 1].min() - 1
        y_max = box[:, 1].max() + 1
        box = [x_min, y_min, x_max, y_max]
        r_boxes.append(box)
    dt_boxes = np.array(r_boxes)
    if dt_boxes is None:
        return None, None
    img_crop_list = []
    for i in range(len(dt_boxes)):
        det_box = dt_boxes[i]
        x0, y0, x1, y1 = expand(2, det_box, ori_im.shape)
        text_rect = ori_im[int(y0):int(y1), int(x0):int(x1), :]
        img_crop_list.append(text_rect)
    rec_res, elapse = text_recognizer(img_crop_list)
    return dt_boxes, rec_res


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_pred_html(pred_structures, matched_index, ocr_contents):
    end_html = []
    td_index = 0
    for tag in pred_structures:
        if '</td>' in tag:
            if td_index in matched_index.keys():
                b_with = False
                if '<b>' in ocr_contents[matched_index[td_index][0]] and len(matched_index[td_index]) > 1:
                    b_with = True
                    end_html.extend('<b>')
                for i, td_index_index in enumerate(matched_index[td_index]):
                    content = ocr_contents[td_index_index][0]
                    if len(matched_index[td_index]) > 1:
                        if len(content) == 0:
                            continue
                        if content[0] == ' ':
                            content = content[1:]
                        if '<b>' in content:
                            content = content[3:]
                        if '</b>' in content:
                            content = content[:-4]
                        if len(content) == 0:
                            continue
                        if i != len(matched_index[td_index]) - 1 and ' ' != content[-1]:
                            content += ' '
                    end_html.extend(content)
                if b_with:
                    end_html.extend('</b>')
            end_html.append(tag)
            td_index += 1
        else:
            end_html.append(tag)
    return ''.join(end_html), end_html


def match_result(dt_boxes, pred_bboxes):
    matched = {}
    for i, gt_box in enumerate(dt_boxes):
        # gt_box = [np.min(gt_box[:, 0]), np.min(gt_box[:, 1]), np.max(gt_box[:, 0]), np.max(gt_box[:, 1])]
        distances = []
        for j, pred_box in enumerate(pred_bboxes):
            distances.append(
                (distance(gt_box, pred_box), 1. - compute_iou(gt_box, pred_box)))  # 获取两两cell之间的L1距离和 1- IOU
        sorted_distances = distances.copy()
        # 根据距离和IOU挑选最"近"的cell
        sorted_distances = sorted(sorted_distances, key=lambda item: (item[1], item[0]))
        if distances.index(sorted_distances[0]) not in matched.keys():
            matched[distances.index(sorted_distances[0])] = [i]
        else:
            matched[distances.index(sorted_distances[0])].append(i)
    return matched


def to_excel(html_table, excel_path):
    ','.join(html_table)
    df = pd.read_html(html_table,index_col=0)
    bb = pd.ExcelWriter(excel_path)
    df[0].to_excel(bb)
    bb.close()


def html_change(html_str):
    result = [x + '>' for x in html_str.split('>')]
    result[-1] = result[-1].strip('>')
    # print(result)
    return result
#
#
#
# # visualization setting
# do_visualize = 1 # whether to visualize
# vis_dir = "demo/table_recognition/lgpma/vis/" # path to save visualization results
#
# # path setting
#
#
#
#
# # loading model from config file and pth file
#
#
# # getting image prefix and test dataset from config file
# img_prefix = model.cfg["data"]["test"]["img_prefix"]
# test_dataset = model.cfg["data"]["test"]["ann_file"]
# with jsonlines.open(test_dataset, 'r') as fp:
#     test_file = list(fp)
#
# # generate prediction of html and save result to savepath
# pred_dict = dict()
# for sample in tqdm(test_file):
#     # predict html of download
#     img_path = img_prefix + sample['filename']
#     result = inference_model(model, img_path)[0]
#     pred_dict[sample['filename']] = result['html']
#
#     # detection results visualization
#     if do_visualize:
#         img = cv2.imread(img_path)
#         img_name = img_path.split("/")[-1]
#         bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] for b in result['bboxes']]
#         for box in bboxes:
#             for j in range(0, len(box), 2):
#                 cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1)
#         cv2.imwrite(vis_dir + img_name, img)
#
#
#
# # generate ground-truth of html from pubtabnet annotation of test dataset.
# gt_dict = dict()
# for data in test_file:
#     if data['filename'] in pred_dict.keys():
#         str_true = data['html']['structure']['tokens']
#         gt_dict[data['filename']] = {'html': format_html(data)}
#
# # evaluation using script from PubTabNet
# teds = TEDS(structure_only=True, n_jobs=16)
# scores = teds.batch_evaluate(pred_dict, gt_dict)
# print(np.array(list(scores.values())).mean())
