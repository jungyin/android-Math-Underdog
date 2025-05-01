from PIL import ImageGrab
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QLabel, QWidget,QShortcut
import ctypes

from pynput import keyboard

# 获取屏幕的缩放比
def get_screen_scaling():
    return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
 
class ScreenshotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.begin = None
        self.end = None
        # Qt.WindowStaysOnTopHint 置顶窗口
        # Qt.FramelessWindowHint 产生一个无窗口边框的窗口，此时用户无法移动该窗口和改变它的大小
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # setWindowOpacity() 只能作用于顶级窗口，取值为0.0~1.0，0.0为完全透明，1.0为完全不透明
        self.setWindowOpacity(0.5)  # 设置窗口透明度为 0.5，如果不加这行代码的话，运行代码后屏幕会被不透明白屏铺满
        self.setWindowState(Qt.WindowFullScreen)  # 铺满全屏幕
        self.screen_scaling = get_screen_scaling()
 
    def mousePressEvent(self, event):
        self.begin = event.pos() 
        self.end = event.pos()
 
    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
 
    # 根据坐标进行截图保存
    def mouseReleaseEvent(self, event):
        self.close()
        # 坐标乘以缩放比后再进行抓取
        x1, y1 = self.begin.x() * self.screen_scaling, self.begin.y() * self.screen_scaling
        x2, y2 = self.end.x() * self.screen_scaling, self.end.y() * self.screen_scaling
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        im = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        # im.show()
        im.save("jietu.jpg")
 
    # 在截图时绘制矩形，目的是为了清楚看到自己所选的区域
    def paintEvent(self, event):
        if not self.begin:
            return
        painter = QPainter(self)  # 创建QPainter对象
        # painter.setPen(Qt.green)
        # painter.setPen(Qt.red) #设置画笔的颜色
        pen = QPen(Qt.red)
        pen.setWidth(2)  # 设置画笔的宽度
        painter.setPen(pen)
 
        # drawRect来绘制矩形，四个参数分别是x,y,w,h
        painter.drawRect(self.begin.x(), self.begin.y(),
                         self.end.x() - self.begin.x(), self.end.y() - self.begin.y())
 
 
def main():
    app = QApplication([])
    widget = ScreenshotWidget()
    widget.show()
# app.exec_()是PyQt中的一个方法，用于启动应用程序的事件循环。它会在调用之后开始监视事件，并根据事件的发生自动执行相应的函数。
    app.exec_()
 
 
if __name__ == '__main__':
    main()