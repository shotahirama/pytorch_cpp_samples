#!/usr/bin/env python
#coding:utf-8

import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()

example = torch.zeros(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")
