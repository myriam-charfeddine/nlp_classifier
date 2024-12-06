import torch
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_ouputs=False):
        labels = inputs.get('labels')

        #Forward Pass
        outputs = model(**inputs)
        logits = outputs.logits

        #Compute Custom Loss
        loss_fn = nn.CrossEntropyLoss(weight= torch.tensor(self.class_weights).to(device=self.device))
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_ouputs else loss
    
    def set_compute_weights(self, class_weights):
        self.class_weights = class_weights

    def set_device(self, device):
        self.device = device
