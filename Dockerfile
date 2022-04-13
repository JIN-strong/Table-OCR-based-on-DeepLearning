FROM ubuntu:20.04
#RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
#COPY ./requirements.txt /requirements.txt
WORKDIR /
#RUN pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
#RUN pip3 install -r requirements.txt
#RUN pip3 install -U https://paddleocr.bj.bcebos.com/whl/layoutparser-0.0.0-py3-none-any.whl
#
#RUN pip install mmcv-full  -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10/index.html
#RUN pip install mmdet
#RUN pip install paddleocr
#RUN pip install opencv-python
#RUN  apt-get update
#RUN  apt install libgl1-mesa-glx
COPY . /
#CMD ["streamlit","run","app.py","--server.port","8502"]
