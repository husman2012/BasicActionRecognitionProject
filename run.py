import torch
import cv2
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
from mmaction.models import build_detector
from mmcv.runner import load_checkpoint
import numpy as np
import mmcv

def add_bboxes(bboxes):
    stdet_bboxes = bboxes
    stdet_bboxes[:, ::2] = stdet_bboxes[:, ::2] 
    stdet_bboxes[:, 1::2] = stdet_bboxes[:, 1::2]
    stdet_bboxes = torch.from_numpy(stdet_bboxes).to(device)
    return stdet_bboxes

def draw_output_frame(bboxes, preds, frame):
    for bbox, pred in zip(bboxes, preds):
        # draw bbox
        box = bbox.astype(np.int64)
        st, ed = tuple(box[:2]), tuple(box[2:])
        cv2.rectangle(frame, st, ed, (0, 0, 255), 2)

        # draw texts
        for k, (label, score) in enumerate(pred):
            if k >= max_labels_per_bbox:
                break
            text = f'{label}: {score:.4f}'
            location = (0 + st[0], 18 + k * 18 + st[1])
            textsize = cv2.getTextSize(text, text_fontface,
                                        text_fontscale,
                                        text_thickness)[0]
            textwidth = textsize[0]
            diag0 = (location[0] + textwidth, location[1] - 14)
            diag1 = (location[0], location[1] + 2)

            cv2.rectangle(frame, diag0, diag1, (0,0,255), -1)
            cv2.putText(frame, text, location, text_fontface,
                        text_fontscale, text_fontcolor,
                        text_thickness, text_linetype)
    return frame

def get_pred_labels(bboxes, action_pred, label_map, std_detboxes):
    preds = []
    for _ in range(std_detboxes.shape[0]):
        preds.append([])

    for class_id in range(len(action_pred)):
        if class_id + 1 not in label_map:
            continue
        for bbox_id in range(std_detboxes.shape[0]):

            if action_pred[class_id][bbox_id, 4] > score_thr:
                preds[bbox_id].append((label_map[class_id + 1],
                                           action_pred[class_id][bbox_id, 4]))
    return preds

#Config Parameters for Models
human_det_config = './demo/faster_rcnn_r50_fpn_2x_coco.py'
human_det_checkpoint = './checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

action_config = Config.fromfile('configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py')
action_checkpoint = 'checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth'
action_config['model']['test_cfg']['rcnn']['action_thr'] = .0

#Assign Device for faster inference
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
cap = cv2.VideoCapture(0)

#Instantiate Human Detector for bounding boxes
human_det_model = init_detector(human_det_config, human_det_checkpoint, device)

#Instantiate Action-Recognition Model and set to Eval
action_model = build_detector(action_config.model, test_cfg =action_config.get('test_cfg'))
load_checkpoint(action_model, action_checkpoint, map_location='cpu')
action_model.to(device)
action_model.eval()

#Set Parameters for Model's Threshold 
score_thr = 0.4
img_norm_cfg = action_config['img_norm_cfg']
if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
    to_bgr = img_norm_cfg.pop('to_bgr')
    img_norm_cfg['to_rgb'] = to_bgr
img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

#Set Parameters for resizing images later
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
display_size = (w, h)
stdet_input_shortside=256
stdet_input_size = mmcv.rescale_size((w, h), (stdet_input_shortside, np.Inf))

#Set up label Map for outputs to frames
label_map_path = 'tools/data/ava/label_map.txt'
with open(label_map_path) as f:
    lines = f.readlines()
lines = [x.strip().split(': ') for x in lines]
label_map = {int(x[0]): x[1] for x in lines}

#Text information
max_labels_per_bbox=5
text_fontface=cv2.FONT_HERSHEY_DUPLEX
text_fontscale=0.5
text_fontcolor=(255, 255, 255) # white
text_thickness=1
text_linetype=1


ret, frame = cap.read()

while ret:
    captured_frames = []
    for i in range(0, 8):
        ret, frame = cap.read()
        frame = mmcv.imresize(frame, display_size)
        processed_frame = mmcv.imresize(frame, stdet_input_size).astype(np.float32)
        _ = mmcv.imnormalize_(processed_frame, **img_norm_cfg)
        resized = cv2.resize(processed_frame, (341, 256))
        captured_frames.append(resized)
    
    bbbox_frame = cv2.resize(frame, (341, 256))
    result = inference_detector(human_det_model, frame)[0]
    bboxes = result[result[:, 4] >= score_thr][:, :4]

    if len(bboxes) <= 0:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    input_array = np.stack(captured_frames).transpose((3, 0, 1, 2))[np.newaxis]

    input_tensor = torch.from_numpy(input_array).to(device).float()

    std_detboxes = add_bboxes(bboxes)

    img_shape  = processed_frame.shape[:2]

    model_input = dict(return_loss = False, 
    img = [input_tensor], 
    proposals = [[std_detboxes]], 
    img_metas = [[dict(img_shape=img_shape)]])

    with torch.no_grad():
        action_pred = action_model(**model_input)[0]
    
    preds = get_pred_labels(bboxes, action_pred, label_map, std_detboxes)

    frame = draw_output_frame(bboxes, preds, frame)
                                        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break