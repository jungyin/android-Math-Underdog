from gevent.pywsgi import WSGIServer

class LongConnectionService:
    def start(self):
        """启动长连接服务"""
        print("Starting long connection service...")
        http_server = WSGIServer(('192.168.10.191', 5001), self.app)
        http_server.serve_forever()

    @property
    def app(self):
        """定义一个简单的应用来处理长连接"""
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/long-connection', methods=['GET'])
        def long_connection():
            while True:
                # 这里只是一个例子，实际中你可能需要某种机制来终止循环或处理数据
                if request.environ.get('werkzeug.server.shutdown'):
                    break
            return jsonify({"status": "Long connection closed"}), 200

        return app