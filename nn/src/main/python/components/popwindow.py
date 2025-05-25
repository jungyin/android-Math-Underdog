import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QShortcut,QSizePolicy
from PyQt5.QtCore import Qt, QRect, QPoint, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView,QWebEngineProfile
from PyQt5.QtGui import QKeySequence
import os
import functools
# 设置环境变量以启用详细的日志记录
# os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--enable-logging --v=1'
# os.environ['QT_LOGGING_RULES'] = 'qt.webengine.*=true'
class FloatingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_position = QPoint()  # 存储拖拽时鼠标相对于窗口左上角的位置
        self.initUI()
 

    def initUI(self):
        # 设置窗口无边框和透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 移除了 Qt.WindowStaysOnTopHint 以便可以正常拖动
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 获取屏幕尺寸并设置窗口大小
        screen_geometry = QApplication.desktop().screenGeometry()
        window_width = screen_geometry.width() // 5
        window_height = screen_geometry.height() // 2
        self.setGeometry(QRect(screen_geometry.width() - window_width, 0, window_width, window_height))

        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)  # 确保没有额外的空间

        # 创建顶部状态栏
        top_bar = QWidget(self)
        top_bar.setObjectName("topBar")  # 可以为状态栏添加样式
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # 添加最小化、固定到顶部和关闭按钮
        minimize_btn = QPushButton("—", self)
        pin_btn = QPushButton("P", self)
        referen_btn = QPushButton("R", self)
        close_btn = QPushButton("X", self)

        minimize_btn.clicked.connect(self.showMinimized)
        pin_btn.clicked.connect(self.toggleAlwaysOnTop)
        referen_btn.clicked.connect(self.referWebView)
        close_btn.clicked.connect(self.close)

        top_layout.addWidget(minimize_btn)
        top_layout.addWidget(pin_btn)
        top_layout.addWidget(referen_btn)
        
        # top_layout.addStrut(1)
        top_layout.addWidget(close_btn)

        main_layout.addWidget(top_bar)

        # 创建并配置QWebEngineView
        self.web_view = QWebEngineView(self)
        self.web_view.setUrl(QUrl("http://localhost:8080/"))  # 替换为你想展示的网址

        # 设置QWebEngineView填充剩余空间
        main_layout.addWidget(self.web_view)
        self.web_view.setContentsMargins(0, 0, 0, 0)
        
        # 设置尺寸策略，使其尽可能扩展
        self.web_view.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

    def bindquick(self,qf,bindquick:dict={}):
         # 快捷键触发显示/隐藏窗口
        show_hide_shortcut = QShortcut(QKeySequence('Ctrl+Shift+Q'), self)
        show_hide_shortcut.activated.connect(self.toggle_visibility)

        for key, value in bindquick.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(value)



    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
    def referWebView(self):
        self.web_view.reload()
    def toggleAlwaysOnTop(self):
        flags = self.windowFlags()
        if flags & Qt.WindowStaysOnTopHint:
            self.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
        self.show()
    def bindtop(self):
        flags = self.windowFlags()
        self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    floating_window = FloatingWindow()
    floating_window.show()
    sys.exit(app.exec_())