from models.sim import NetworkLight
import torch
import torch.nn as nn
from .baseline import BaselineModel
import torchvision.models as models


class Regressor(BaselineModel):
    def __init__(self, n_classes, **kwargs):
        super(Regressor, self).__init__(**kwargs)
        self.model = NetworkLight()
        self.name = 'NetworkLight'
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
        self.set_optimizer_params()
        self.n_classes = n_classes

        if self.device is not None:
            self.model.to(self.device)
            self.criterion.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["categories"]

        if self.device is not None:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def inference_step(self, batch):
        inputs = batch['imgs']
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)

        if self.device:
            preds = preds.cpu()
        return preds.numpy()

    def evaluate_step(self, batch):
        inputs = batch["imgs"]
        targets = batch["categories"]

        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self(inputs)  # batch_size, category_dim
        loss = self.criterion(outputs, targets)

        metric_dict = self.update_metrics(outputs, targets)

        return loss, metric_dict

    def inference_img(self, img):
        self.model.eval()
        with torch.no_grad():
            if img.shape[0] != 1:
                img = img.unsqueeze(0)
            if self.device is not None:
                img = img.to(self.device)

            outputs = self(img)

            if self.device is not None:
                outputs = outputs.cpu()

        return outputs.squeeze(0).numpy()

    def forward_test(self):
        inputs = torch.rand(1, 3, 224, 224)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs
