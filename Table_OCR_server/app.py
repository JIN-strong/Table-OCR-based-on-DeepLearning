import os

os.environ['NUMEXPR_MAX_THREADS'] = '20'
import sys

sys.path.append('./')
sys.path.append('./model/')
import RARE.ppstructure.table.predict_table as RARE
import LGPMA.demo.table_recognition.lgpma.tools.lgpma as LPGMA
import layoutparser as lp
import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import numpy as np
import cv2
# 画图
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from math import ceil

# define args
arg_dic_pd = {
    "det_model_dir": "model/RARE/inference/ch_PP-OCRv2_det_infer",
    "rec_model_dir": "model/RARE/inference/ch_PP-OCRv2_rec_infer",
    "table_model_dir": "model/RARE/inference/en_ppocr_mobile_v2.0_table_structure_infer",
    "rec_char_dict_path": "model/RARE/ppocr/utils/ppocr_keys_v1.txt",
    "table_char_dict_path": "model/RARE/ppocr/utils/dict/table_structure_dict.txt",
    "output": "output/",
    "image_dir": "tmp/"
}
# arg_dic_pd["image_dir"] = "tmp/"

arg_dic_lp = {
    "det_model_dir": "model/RARE/inference/ch_PP-OCRv2_det_infer",
    "rec_model_dir": "model/RARE/inference/ch_PP-OCRv2_rec_infer",
    "rec_char_dict_path": "model/RARE/ppocr/utils/ppocr_keys_v1.txt",
    "checkpoint_file": "model/LGPMA/demo/table_recognition/datalist/maskrcnn-lgpma-pub-e12-pub.pth",
    "config_file": "model/LGPMA/demo/table_recognition/lgpma/configs/lgpma_pub.py",
    "output": "output/",
    "image_dir": "tmp/"
}


# arg_dic_lp["image_dir"] = "tmp/"

@st.cache(allow_output_mutation=True)
def get_manager():
    return stx.CookieManager()


@st.cache
def make_dir(cookie):
    os.makedirs('output/' + cookie + "/up", exist_ok=True)
    os.makedirs('output/' + cookie + "/down", exist_ok=True)


@st.cache
def layout_rec(image, threshold_):
    image = image[..., ::-1]
    # 加载模型
    model = lp.PaddleDetectionLayoutModel(config_path="lp://TableBank/ppyolov2_r50vd_dcn_365e_tableBank_word/config",
                                          threshold=threshold_,
                                          label_map={0: "Table"},
                                          enforce_cpu=False,
                                          enable_mkldnn=True)
    # 检测
    layout = model.detect(image)
    table_blocks = lp.Layout([b for b in layout if b.type == 'Table'])
    # 显示结果
    show_img = lp.draw_box(image, table_blocks, box_width=3, show_element_type=True, show_element_id=True)
    idx_ = None
    for idx_, ele in enumerate(table_blocks):
        points = ele.points
        p1 = points[0]
        p2 = points[2]
        im = image[int(p1[1]) - 5:ceil(p2[1]) + 5, int(p1[0]):ceil(p2[0])]
        cv2.imwrite('output/' + cookies + "/up/" + str(idx_) + '.png', im)
    return show_img, idx_


@st.cache
def table_rec_RARE(uploaded_file):
    arg_dic_pd["output"] = 'output/' + cookies + "/down/"
    arg_dic_pd["image_dir"] = 'output/' + cookies + "/up/"

    index_save = RARE.pd_ocr(arg_dic=arg_dic_pd)
    return index_save


@st.cache
def table_rec_lg(uploaded_file, index):
    arg_dic_lp["output"] = 'output/' + cookies + "/down/"
    arg_dic_lp["image_dir"] = 'output/' + cookies + "/up/"
    LPGMA.lgpma_ocr(index, args_dic=arg_dic_lp)


@st.cache
def load_local_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    image = np.array(Image.open(BytesIO(bytes_data)))
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image, image_cv2


# 构建基本界面
cookie_manager = get_manager()
cookies = cookie_manager.get_all()

if cookies:
    cookies = cookies["_hp2_id.1823968819"]["sessionId"]
    make_dir(cookies)
    st.title('A tool for ble recognition')
    st.text("Welcome to this website !")
    st.subheader('First, please upload an image')
    width = st.sidebar.slider("picture width", 200, 700, 500)
    threshold = st.sidebar.slider("table detection threshold", 0.1, 1.0, 0.5)

    # 上传文件
    uploaded_pic = st.file_uploader("Choose a picture", type=['png', 'jpg', 'bmp'])

    if uploaded_pic is not None:
        img, img_cv2 = load_local_image(uploaded_pic)
        st.image(img, caption='Preview of input', width=width, channels="RGB")
        idx = None
        show_image, idx = layout_rec(img_cv2, threshold)
        if idx is not None:
            st.text('Now, the table in the picture has been detected.')
            st.image(show_image, caption='Table detection results', width=width, channels="RGB")

            st.subheader('Second, choose the model and download serial number')
            model_choose = st.selectbox('Model', ('RARE', 'LPGMA'))
            if model_choose == 'RARE':
                index = table_rec_RARE(uploaded_pic)
            else:
                index = table_rec_RARE(uploaded_pic)
                table_rec_lg(uploaded_pic, index)
            # 输出结果

            st.subheader('Third, check and download the results')
            st.text('There may be differences in the preview results, please download the excel file to view!')
            for i in range(0, idx + 1):
                excel_path = 'output/' + cookies + "/down/" + str(i) + ".xlsx"
                df = pd.read_excel(excel_path)
                df = df.astype(str)
                st.table(df)

                with open(excel_path, "rb") as fp:
                    st.download_button(
                        label="Download Output Excel No." + str(i),
                        data=fp,
                        file_name="table" + str(i) + ".xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        else:
            st.error(
                'Sorry, the table has not been detected in the picture, please adjust the table detection threshold!')
