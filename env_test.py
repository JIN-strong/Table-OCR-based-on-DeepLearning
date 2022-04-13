import os
os.environ['NUMEXPR_MAX_THREADS'] = '20'
import model.RARE.ppstructure.table.predict_table as RARE
import model.LGPMA.demo.table_recognition.lgpma.tools.lgpma as LPGMA
import cv2
import layoutparser as lp

# arg_dic_pd = {
#     "det_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_det_infer",
#     "rec_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_rec_infer",
#     "table_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_structure_infer",
#     # "image_dir": "model/RARE/doc/download/download.jpg",
#     "rec_char_dict_path": "model/RARE/ppocr/utils/dict/table_dict.txt",
#     "table_char_dict_path": "model/RARE/ppocr/utils/dict/table_structure_dict.txt",
#     "output": "output/download/"
# }

arg_dic_lp = {
    "det_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_det_infer",
    "rec_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_rec_infer",
    "rec_char_dict_path": "model/RARE/ppocr/utils/dict/table_dict.txt",
    "checkpoint_file": "model/LGPMA/demo/table_recognition/datalist/maskrcnn-lgpma-pub-e12-pub.pth",
    "config_file": "model/LGPMA/demo/table_recognition/lgpma/configs/lgpma_pub.py",
    "output": "output/"
    # "image_dir": "model/RARE/doc/download/download.jpg"
}

# image = cv2.imread("model/RARE/doc/table/1.png")
# # 加载模型
# model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
#                                       threshold=0.5,
#                                       label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
#                                       enforce_cpu=False,
#                                       enable_mkldnn=True)
# # 检测
# layout = model.detect(image)
# table_blocks = lp.Layout([b for b in layout if b.type == 'Table'])
#
# # 显示结果
# show_img = lp.draw_box(image, table_blocks, box_width=3, show_element_type=True, show_element_id=True)
# show_img.save('out.png')
# # show_img.show()
#
# # ai jian
# for idx, ele in enumerate(table_blocks):
#     points = ele.points
#     p1 = points[0]
#     p2 = points[2]
#     im = image[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
#     cv2.imwrite("tmp/" + str(idx) + '.png', im)

arg_dic_lp["image_dir"] = "tmp/"
LPGMA.lgpma_ocr(args_dic=arg_dic_lp)

# arg_dic_pd["image_dir"] = "tmp/"
# RARE.pd_ocr(arg_dic=arg_dic_pd)
