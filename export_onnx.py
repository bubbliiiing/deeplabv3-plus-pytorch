#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# by [jackhanyuan](https://github.com/jackhanyuan)

import onnx
import torch
from pathlib import Path
from nets.deeplabv3_plus import DeepLab

num_classes = 21
input_shape = [512, 512]
backbone = "mobilenet"  # backbone：mobilenet、xception
downsample_factor = 8  # downsample factor, same with training
pretrained = False

cuda = False    # use cuda
train = False  # model.train() mode
simplify = True  # simplify onnx
model_path = 'model_data/deeplab_mobilenetv2.pth'  # *.pth model path


def convert_to_onnx(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained)
    model_path = Path(model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    if cuda:
        model = model.cuda()
    print('{} loaded.'.format(model_path))

    im = torch.zeros(1, 3, *input_shape).to(device)  # image size(1, 3, 512, 512) BCHW
    input_layer_names = ["images"]
    output_layer_names = ["output"]

    # Update model
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction

    # Export the model
    print(f'Starting export with onnx {onnx.__version__}.')
    model_path = model_path.with_suffix('.onnx')
    torch.onnx.export(model,
                      im,
                      f=model_path,
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=not train,
                      input_names=input_layer_names,
                      output_names=output_layer_names,
                      dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(model_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify onnx
    if simplify:
        import onnxsim
        print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, model_path)

    print('Onnx model save as {}'.format(model_path))


if __name__ == '__main__':
    convert_to_onnx(model_path)