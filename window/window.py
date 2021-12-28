# -*- coding: utf-8 -*-

import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil

# import cv2
# import tensorflow.keras as keras
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import test

class MainWindow(QTabWidget):
    # 初始化
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/logo.png'))
        self.setWindowTitle('变电站识别系统')  # todo 修改系统名称
        # 模型初始化
        self.to_predict_name = "images/start.jpg"  # todo 修改初始图片，这个图片要放在images目录下
        self.resize(900, 700)
        self.initUI()

    # 界面初始化，设置界面布局
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' 变电站识别 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 16))
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        # 关于页面，设置组件并把组件放在布局上
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用变电站识别系统')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/bj.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel("作者：黄伟群")  # todo 更换作者信息
        label_super.setFont(QFont('楷体', 12))
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        # 添加注释
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    # 上传并显示图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')  # 打开文件选择框选择文件
        img_name = openfile_name[0]  # 获取图片名称
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_up." + img_name.split(".")[-1]  # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)  # 打开图片
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到400的高，方便界面显示
            cv2.imwrite("images/show.jpg", img_show)
            # img_init = cv2.resize(img_init, (224, 224))  # 将图片大小调整到224*224用于模型推理
            # cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.jpg"))
            self.result.setText("等待识别")

    # 预测图片
    def predict_img(self):
        # img = Image.open('images/target.png')  # 读取图片
        # img = np.asarray(img)  # 将图片转化为numpy的数组
        # outputs = self.model.predict(img.reshape(1, 224, 224, 3))  # 将图片输入模型得到结果
        # result_index = int(np.argmax(outputs))
        # result = self.class_names[result_index]  # 获得对应的名称
        # self.result.setText(result)  # 在界面上做显示
        version = 8
        test_img = test.imread('images/show.jpg')
        out_mask = np.zeros((test_img.shape[0], test_img.shape[1]))

        img_x, img_y = test_img.shape[0], test_img.shape[1]

        # test_model = keras.models.load_model('/kaggle/working/best_model_v%d.h5'%version)
        test.model.load_weights('./best_model_v%d.h5' % version)
        image = np.zeros((1536, 1536, 3))
        image[:img_x, :img_y] = test_img[:img_x, :img_y]
        image = test.get_preprocessing(preprocessing_fn=test.preprocess_input)(image=image)['image']
        image = np.expand_dims(image, axis=0)
        out_mask = test.model.predict(image)[0, :img_x, :img_y, 0]

        test.visualize(
            image=test.denormalize(test_img.squeeze()),
            out_mask=out_mask,
            out_mask_rounded=out_mask.round()
        )

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    x.predict_img()
    sys.exit(app.exec_())
