import os
from flask import Flask
from wsgiref import simple_server
from flask_cors import cross_origin, CORS

from NewsArticleSorting.NASLogger import logging

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])
def index():
    logging.info("Testing logging info")
    return "Initial setup for News Article Sorting Application"


port = int(os.getenv("PORT", 5000))

if __name__=="__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
