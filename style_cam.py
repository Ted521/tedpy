import argparse
import os
import sys
import time
import re
import cv2

import torch
from torchvision import transforms
from transformer_net import TransformerNet

def camapply(frame, device, style_model):
    content_image = frame
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    output = style_model(content_image).cpu()
    img = output[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return img

def main():
    device = torch.device("cuda")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model for stylizing")

    args = parser.parse_args()
    
    cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture('test.avi') # for vid
    w = round(cam.get(3))
    h = round(cam.get(4))
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    now = time.localtime()
    f_strnow = time.strftime("%Y%m%d_%H%M", now)
    out = cv2.VideoWriter(f_strnow+'_'+args.model.split('/')[-1].split('.')[0]+'.avi', fourcc, fps, (w,h))
    
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break
            
            output = camapply(frame, device, style_model)
            cv2.imshow('Result', output)
            out.write(output)
            if cv2.waitKey(1) == ord('q'):
                break
    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
