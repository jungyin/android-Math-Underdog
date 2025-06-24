from components.popwindow import FloatingWindow
from components.screenshot import ScreenshotWidget
from components.global_key import GlobalKeyListener
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
from transformers import AutoTokenizer
from infer.latex_ocr.openvino_infer import LatexMoelRun
import keyboard
import sys
import time
import pyperclip

class QuickTools(QObject):
    showNewPageSignal = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        p = "./assets/latex_ocr/"
        ocr_model = LatexMoelRun(p)
        tokenizer = tokenizer = AutoTokenizer.from_pretrained(p, max_len=296)

        self.math_ocr_model = ocr_model
        self.math_ocr_token = tokenizer
        self.screenshot = None

        
        self.showNewPageSignal.connect(self.pyqtHandler)

    def img_callback(self,im):
        ocrstr_ids = self.math_ocr_model.greedy_search(im)
       
        ocrstr:str = self.math_ocr_token.decode(ocrstr_ids)
        

        if(ocrstr.startswith("<s><s>") and ocrstr.endswith("</s>")):
                ocrstr = self.math_ocr_model.process_mixed_content(ocrstr)[0]
                pyperclip.copy(ocrstr)
        else:
            pyperclip.copy("无法识别该公式")

        print(ocrstr)
        self.screenshot.close()
        self.screenshot = None

        self.popwindow.bindtop()

    def ocr_screenshot(self):
        self.screenshot = ScreenshotWidget(self.img_callback)
        self.screenshot.clipFrame = None
        self.screenshot.show()
    def pyqtHandler(self,value):
        if(value == 1):
            self.ocr_screenshot()

    def bindpop(self,popwindow:FloatingWindow):
        self.popwindow = popwindow





if __name__ == '__main__':

    app = QApplication(sys.argv)
    
    tool = QuickTools()

    QT_keylist={}

    ALL_keylist={"Ctrl+Shift+A":lambda: tool.showNewPageSignal.emit(1)}


    listener_thread = GlobalKeyListener(ALL_keylist)
    listener_thread.start()

    def quick_function(key:str):
        if(key=='math_ocr'):
            work_str:str = tool.ocr_screenshot()
            if(work_str.startswith("<s><s>") and work_str.endswith("</s>")):
                pyperclip.copy(work_str)
            else:
                pyperclip.copy("无法识别该公式")

            print(work_str)
        else:
            pass

    floating_window = FloatingWindow()
    floating_window.bindquick(quick_function,QT_keylist)
    floating_window.show()

    tool.bindpop(floating_window)
    sys.exit(app.exec_())