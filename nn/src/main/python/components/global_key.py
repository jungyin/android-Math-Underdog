import keyboard
from PyQt5.QtCore import QThread, pyqtSignal






class GlobalKeyListener(QThread):
    def __init__(self, hotkeys):
        super().__init__()
        self.hotkeys = hotkeys

    def run(self):
       
        # 注册快捷键
        for key, func in self.hotkeys.items():
            keyboard.add_hotkey(key, func,suppress=True)

        # 进入无限循环等待，保持线程存活
        keyboard.wait()  # 注意: 这里会阻塞当前线程