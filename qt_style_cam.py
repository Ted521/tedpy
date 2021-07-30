# coding = utf-8
"""
Created by Ted at 2021-07-28
"""

import cv2
import sys # , random
import re
from PyQt5 import QtCore, QtWidgets, QtGui
import warnings
import torch
from torchvision import transforms
from transformer_net import TransformerNet
# import time
import os

warnings.filterwarnings("ignore")


class ShowVideo(QtCore.QObject):
    flag = 0
    st_flag = 0

    cam = cv2.VideoCapture(2)
    ret, image = cam.read()
    height, width = image.shape[:2]

    fps = int(cam.get(cv2.CAP_PROP_FPS))
    f_dir = 'saved_models'
    f_list = os.listdir(f_dir)

    r = 0
    # r = random.randint(0, 3)
    style_model = TransformerNet()

    # state_dict = torch.load('saved_models/'+f_list[r])
    # for k in list(state_dict.keys()):
    #     if re.search(r'in\d+\.running_(mean|var)$', k):
            # del state_dict[k]

    device = torch.device("cuda")

    # style_model.load_state_dict(state_dict)
    # style_model.to(device)
    # print('model loaded')

    vidsignal1 = QtCore.pyqtSignal(QtGui.QImage)
    vidsignal2 = QtCore.pyqtSignal(QtGui.QImage)
    vidsignal3 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)
        self.loadmodel()

    def loadmodel(self):
        state_dict = torch.load('saved_models/'+self.f_list[self.r])
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        self.style_model.load_state_dict(state_dict)
        self.style_model.to(self.device)

    @QtCore.pyqtSlot()
    def startVideo(self):
        global image

        run_video = True

        while run_video:
            ret, image = self.cam.read()
            color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qt_image1 = QtGui.QImage(color_swapped_image.data,
                                     self.width,
                                     self.height,
                                     color_swapped_image.strides[0],
                                     QtGui.QImage.Format_RGB888)
            self.vidsignal1.emit(qt_image1)

            if self.flag:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_canny = cv2.Canny(img_gray, 50, 100)
                qt_image2 = QtGui.QImage(img_canny.data,
                                         self.width,
                                         self.height,
                                         img_canny.strides[0],
                                         QtGui.QImage.Format_Grayscale8)
                self.vidsignal2.emit(qt_image2)

            if self.st_flag:
                with torch.no_grad():
                    content_image = image
                    content_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))
                                    ])
                    content_image = content_transform(content_image)
                    content_image = content_image.unsqueeze(0).to(self.device)
                    output = self.style_model(content_image).cpu()
                    img = output[0].clone().clamp(0, 255).numpy()
                    img = img.transpose(1, 2, 0).astype("uint8")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    qt_image3 = QtGui.QImage(img.data,
                                             self.width,
                                         self.height,
                                         img.strides[0],
                                         QtGui.QImage.Format_RGB888)
                    self.vidsignal3.emit(qt_image3)

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit)
            loop.exec_()

    @QtCore.pyqtSlot()
    def canny(self):
        self.flag = 1 - self.flag

    @QtCore.pyqtSlot()
    def style_transfer(self):
        self.st_flag = 1 - self.st_flag

    @QtCore.pyqtSlot()
    def mosaic(self):
        self.r = 0
        self.loadmodel()

    @QtCore.pyqtSlot()
    def candy(self):
        self.r = 1
        self.loadmodel()

    @QtCore.pyqtSlot()
    def rain_princess(self):
        self.r = 2
        self.loadmodel()

    @QtCore.pyqtSlot()
    def udnie(self):
        self.r = 3
        self.loadmodel()


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('TEST')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("V D F")

        self.image = image

        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


def main():
    app = QtWidgets.QApplication(sys.argv)

    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    image_viewer2 = ImageViewer()
    image_viewer3 = ImageViewer()

    vid.vidsignal1.connect(image_viewer1.setImage)
    vid.vidsignal2.connect(image_viewer2.setImage)
    vid.vidsignal3.connect(image_viewer3.setImage)

    push_button1 = QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Canny')
    push_button4 = QtWidgets.QPushButton('Style')
    push_button3 = QtWidgets.QPushButton('Quit')
    push_button6 = QtWidgets.QPushButton('Style 1')
    push_button7 = QtWidgets.QPushButton('Style 2')
    push_button8 = QtWidgets.QPushButton('Style 3')
    push_button9 = QtWidgets.QPushButton('Style 4')
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.canny)
    push_button4.clicked.connect(vid.style_transfer)
    push_button3.clicked.connect(QtCore.QCoreApplication.instance().quit)
    push_button6.clicked.connect(vid.mosaic)
    push_button7.clicked.connect(vid.candy)
    push_button8.clicked.connect(vid.rain_princess)
    push_button9.clicked.connect(vid.udnie)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout2 = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    horizontal_layout.addWidget(image_viewer2)
    horizontal_layout.addWidget(image_viewer3)
    horizontal_layout2.addWidget(push_button6)
    horizontal_layout2.addWidget(push_button7)
    horizontal_layout2.addWidget(push_button8)
    horizontal_layout2.addWidget(push_button9)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)
    vertical_layout.addWidget(push_button4)
    vertical_layout.addLayout(horizontal_layout2)
    vertical_layout.addWidget(push_button3)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
