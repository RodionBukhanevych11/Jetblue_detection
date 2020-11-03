from flask import Flask, jsonify, request
import torchvision
import torch
import os
import requests
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from load_images import createDataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args_dict = {'num_classes': 2, 'in_features': 1024, 'detection_threshold': 0.999,
             'weights':'model2_fasterrcnn_resnet50_weights.pt'}


def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    model.roi_heads.box_predictor = FastRCNNPredictor(args_dict['in_features'], args_dict['num_classes'])
    location = os.environ["MODEL_WEIGHTS"]
    model.load_state_dict(torch.load(location))
    return model

app = Flask(__name__)
model=load_model()
model.to(device)

@app.route("/")
def status():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    img_url = request.args.get('image_url')
    response=requests.get(img_url)
    test_loader = createDataset(response)
    images = next(iter(test_loader))
    image = list(images[0].unsqueeze(0).to(device))
    model.eval()
    outputs = model(image)
    scores = outputs[0]['scores'].data.cpu().numpy()
    scores = scores[scores >= args_dict['detection_threshold']]
    if len(scores) > 0:
        prediction='JetBlue'
    else:
        prediction='Not a JetBlue'

    return jsonify({"image": img_url, "prediction": prediction})


if __name__ == '__main__':
    app.run(host=os.environ["JETBLUE_HOST"], port=os.environ["JETBLUE_PORT"])