# -*- coding:utf-8 -*-
import os
import onnx
import torch
import argparse
from onnxconverter_common import float16

from fastgesture.utils.checkpoint import ckpt_load


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='weights path')
    parser.add_argument('--img-size', type=int, default=160, help='image size')
    parser.add_argument('--save_path', type=str, default='./export_model', help='path to save export onnx file')
    parser.add_argument('--save_name', type=str, default='fg_model', help='new file name')
    args = parser.parse_args()
    
    print(f"Export Config: ", args)
    
    # check model -------------------------------
    ckeckpoint = torch.load(args.weights)
    model, pncs = ckpt_load(model_path=ckeckpoint, export=True)
    model.to('cpu')
    model.eval()
    
    # generate save path ------------------------
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)
    fp32_model_path = args.save_path + "/" + args.save_name + '_fp32.onnx'
    fp16_model_path = args.save_path + "/" + args.save_name + '_fp16.onnx'
    
    # export fp32 model -------------------------
    print(f"Export Onnx Model With FP32 -----------------------------")
    
    input_shape = torch.randn(1, 1, args.img_size, args.img_size).type(torch.FloatTensor)
    torch.onnx.export(
        model,
        input_shape,
        fp32_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        export_params=True,
        do_constant_folding=True,
    )
    
    # export fp16 model -------------------------
    print(f"Export Onnx Model With FP16 -----------------------------")
    
    fp32_onnx = onnx.load(fp32_model_path)
    onnx.checker.check_model(fp32_onnx)
    model_fp16 = float16.convert_float_to_float16(fp32_onnx)
    onnx.save(model_fp16, fp16_model_path)

    # print data --------------------------------
    print(f"Finish --------------------------------------------------")
    
    print(f"PNCS Data: \n{pncs} \n")
    print(f"fp32 onnx model path: {fp32_model_path} \n")
    print(f"fp16 onnx model path: {fp16_model_path} \n")
    