import os
import shutil
from NewsArticleSorting.NASLogger import get_log_list
from flask import Flask, url_for, render_template, request,send_file
from wsgiref import simple_server
from flask_cors import cross_origin, CORS
from NewsArticleSorting.NASEntity.NASArtifactEntity import PredictorArtifact

from NewsArticleSorting.NASLogger import logging
from NewsArticleSorting.NASPipeline.NASPredictionPipeline import NASPredictionPipeline
from NewsArticleSorting.NASPipeline.NASSingleSentencePredictionPipeline import NASSingleSentencePredictionPipeline

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def index():
    project_logo_url = url_for('static', filename='newspaper-computer-icons-symbol-news-icon-png-clip-art.png')
    ineuron_logo_url = url_for('static', filename='ineuron-logo.webp')
    if request.method == 'POST':
        prediction_pipeline = NASSingleSentencePredictionPipeline()
        sentence = request.form["news_text"]
        result = prediction_pipeline.initiate_single_sentence_prediction_pipeline(sentence=sentence)
        return render_template('index2.html', project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url,
                                            result=result)
        
    else:
        logging.info("Testing logging info")
        return render_template('index2.html', project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)


@app.route("/prediction", methods=["POST"])
@cross_origin()
def prediction_route():
    project_logo_url = url_for('static', filename='newspaper-computer-icons-symbol-news-icon-png-clip-art.png')
    ineuron_logo_url = url_for('static', filename='ineuron-logo.webp')

    try:
        if request.form is not None:
            file_item = request.files["dataset"]

            if file_item.filename:
                if os.path.isdir("UploadedFiles"):
                    shutil.rmtree("UploadedFiles")
                    os.mkdir("UploadedFiles")
                else:
                    os.mkdir("UploadedFiles")
                with open(os.path.join("UploadedFiles", file_item.filename), 'wb') as f:
                    f.write(file_item.read())

            prediction_pipeline = NASPredictionPipeline(uploaded_dataset_dir="UploadedFiles")
            artifact: PredictorArtifact = prediction_pipeline.complete_prediction_pipeline()
            render_template("index2.html", message=artifact.message, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)
            
            return send_file(artifact.prediction_result_path, as_attachment=True)    

    except ValueError as e:
        message = f"Value Error: {str(e)}\nTry Again"
        return render_template("index2.html", message=message, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)

    except KeyError as e:
        message = f"Key Error: {str(e)}\nTry Again"
        return render_template("index2.html", message=message, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)

    except Exception as e:
        message = f"Error: {str(e)}\nTry Again"
        return render_template("index2.html", message=message, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)

@app.route("/logs", methods=["GET", "POST"])
@cross_origin()
def get_logs():
    try:
        project_logo_url = url_for('static', filename='newspaper-computer-icons-symbol-news-icon-png-clip-art.png')
        ineuron_logo_url = url_for('static', filename='ineuron-logo.webp')
        if request.method == 'POST':
            if int(request.form["num_logs"]) is not int:
                num_logs = int(request.form["num_logs"])
                logs = get_log_list(num_logs=num_logs)
            else: 
                logs = get_log_list()

            
            return render_template("logs.html", logs=logs, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)
        else:
            return render_template("logs.html", project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)

    except Exception as e:
        message = f"Error: {str(e)}\nTry Again"
        return render_template("logs.html", message=message, project_logo_url=project_logo_url,
                                            ineuron_logo_url=ineuron_logo_url)

port = int(os.getenv("PORT", 5000))

if __name__=="__main__":
    host = '0.0.0.0'
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
    # app.run(debug=True)
