import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from app.detector import Category

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Remove batch dimension if present
        if x.dim() == 4:
            x = x.squeeze(0)
        # Model returns (Losses, Detections) in scripting mode
        _, detections = self.model([x])
        return detections[0]

# Build model
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(Category.classes()))

# Load checkpoint
checkpoint = torch.load("app/models/fasterrcnn_token_detector.pth", map_location="cpu")
# Always use model_state since that's how we save it in training
model.load_state_dict(checkpoint["model_state"])

model.eval()

# Script the wrapped model
wrapped_model = ModelWrapper(model)
scripted_model = torch.jit.script(wrapped_model)
scripted_model.save("app/models/fasterrcnn_token_detector_scripted.pt")
print("Saved TorchScript model.")