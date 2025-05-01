from flask import Flask, request
import handlers.request_handlers as net_interface
from services.long_connection_service import LongConnectionService

app = Flask(__name__)



    
@app.route("/getModelList",methods = ['POST'])
def modelList():
    return net_interface.llm_models(request)

@app.route("/selectModel",methods = ['POST'])
def selectModel():
    return net_interface.llm_select(request)

@app.route("/restartAll",methods = ['POST'])
def restartAll():
    return net_interface.restart_all(request)

@app.route("/stopSpeak",methods = ['POST'])
def stopSpeak():
    return net_interface.stop_speak(request)

@app.route("/getLastStr",methods = ['POST'])
def getLastStr():
    return net_interface.get_llm_out(request)

@app.route("/getMessageList",methods = ['POST'])
def getMessageList():
    return net_interface.get_llm_history(request)

@app.route("/setSystemContext",methods = ['POST'])
def setSystemContext():
    return net_interface.set_system_context(request)

@app.route("/speek",methods = ['POST'])
def speek():
    return net_interface.speek(request)

if __name__ == '__main__':
    # app.run(host='192.168.10.191', port=5090)
    app.run(host='localhost', port=5090)
