import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget,QVBoxLayout,QPushButton
from PyQt5.QtCore import Qt,QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import keyboard

class FloatingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.setGeometry(100, 100, 480, 600)  # 设置窗口大小和位置
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(0)  # 控制部件之间的间距
        layout.setContentsMargins(50, 50, 50, 50)  # 控制布局与窗口边缘的距离

        self.webView = QWebEngineView(self)
        self.setFixedSize(480,600)
        self.webView.load(QUrl("https://www.baidu.com"))  # 加载你的目标网页
        self.webView.show()
        
        # 创建按钮组件
        # button = QPushButton("这是一个按钮", self)
        # button.clicked.connect(self.on_button_clicked)  # 连接按钮点击事件

        # 将WebView和按钮添加到布局中，并设置拉伸因子以均分空间
        layout.addWidget(self.webView, 1)  # 第二个参数是拉伸因子
        # layout.addWidget(button, 1)
        
        self.setCentralWidget(central_widget)

        self.show()

    def mousePressEvent(self, event):
        self.offset = event.pos()
    
    def mouseMoveEvent(self, event):
        x = event.globalX()
        y = event.globalY()
        self.move(x - self.offset.x(), y - self.offset.y())

def toggle_window(window):
    if window.isVisible():
        window.webView.setUrl(QUrl())  # 清理网页内容
        window.hide()
    else:
        window.webView.setUrl(QUrl("https://www.baidu.com"))  # 重新加载网页
        window.show()

app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(False)
window = FloatingWindow()

# 绑定快捷键，例如Ctrl+Shift+W来切换窗口显示
keyboard.add_hotkey('ctrl+w', lambda: toggle_window(window))

sys.exit(app.exec_())