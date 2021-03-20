import tornado.web
import tornado.ioloop
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tornado.options import define, options

define("port", default=8888, help="运行端口", type=int)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        a = self.request.files
        print(a.keys())
        files = self.request.files['files']
        # print(type(files[0]))
        for file in files:
            print(file.keys())
            print(file['filename'])
            img = file['body']
            img = Image.open(BytesIO(img)).convert('RGB')
            plt.imshow(img)
            plt.show()


if __name__ == "__main__":
    app = tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/upload", UploadHandler),
        ],
    )
    app.listen(options.port)
    print("http://localhost:{}/".format(options.port))
    tornado.ioloop.IOLoop.current().start()