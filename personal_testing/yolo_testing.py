from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
# Load the weights from our repository
model_path = hf_hub_download(local_dir=".",
                             repo_id="armvectores/yolov8n_handwritten_text_detection",
                             filename="best.pt")
model = YOLO(model_path)
# Load test blank
test_blank_path = "/Users/conormcgartoll/Documents/Github/lerobot/outputs/yolo_training_images/camera_03_frame_000000.png"
# Do the predictions
res = model.predict(source=test_blank_path, project='.',name='detected', exist_ok=True, save=True, show=False, show_labels=False, show_conf=False, conf=0.5, )
plt.figure(figsize=(15,10))
plt.imshow(plt.imread('detected/test_blank.png'))
plt.show()