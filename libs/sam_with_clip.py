import pathlib
import os.path as osp
from PIL import Image

import torch
import numpy as np
import cv2
import imutils
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import clip


def load_image(img_path, mode='RGB'):
    img = cv2.imread(str(img_path))
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calc_mask_iou(m1, m2):
    m1_seg, m2_seg = m1['segmentation'], m2['segmentation']
    intersection = np.logical_and(m1_seg, m2_seg)
    intersection_area = intersection.astype(int).sum()
    union = np.logical_or(m1_seg, m2_seg)
    union_area = union.astype(int).sum()

    iou = intersection_area / union_area
    m1_overlap = intersection_area / m1['area']
    m2_overlap = intersection_area / m2['area']
    x1_min, y1_min, x1_max, y1_max = m1['bbox']
    x2_min, y2_min, x2_max, y2_max = m2['bbox']
    return {'iou': iou,
            'm1_overlap': m1_overlap,
            'm2_overlap': m2_overlap,
            'intersection': intersection,
            'intersection_area': intersection_area,
            'union': union,
            'union_area': union_area,
            'union_box': (min(x1_min, x2_min), min(y1_min, y2_min), max(x1_max, x2_max), max(y1_max, y2_max))}


def select_largest_contour(input_m):
    m = input_m['segmentation'].astype(np.uint8)
    cnts = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    h, w = m.shape[:2]
    new_m = np.zeros((h, w), dtype=np.uint8)
    new_m = cv2.drawContours(new_m, cnts, 0, color=255, thickness=-1)
    area = cv2.contourArea(cnts[0])
    x, y, w, h = cv2.boundingRect(cnts[0])

    return new_m, area, (x, y, x + w, y + h)


def cut_out(masks: list, image: np.ndarray, pad=10, make_square_crop=False):
    cut_out = []
    for m in masks:
        mask = m['segmentation']
        ys, xs = np.where(mask > 0)
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        if make_square_crop:
            w, h = x_max - x_min, y_max - y_min
            if w > h:
                pad_x, pad_y = 0, (w - h) // 2,
            else:
                pad_x, pad_y = (h - w) // 2, 0
        else:
            pad_x, pad_y = 0, 0

        cut_out.append(image[max(0, y_min - pad_y - pad): y_max + pad_y + pad,
                             max(0, x_min - pad_x - pad): x_max + pad_x + pad])
    return cut_out


class SAMWithCLIP:
    def __init__(self,
                 sam_checkpoint="sam_weights/sam_vit_b_01ec64.pth",
                 sam_model_type="vit_b",
                 sam_predictor_type='auto',      # ['auto', 'points']
                 point_coords=None,
                 device="cuda",
                 crop=None,
                 min_area=0.0005,
                 max_area=0.5,
                 iou_thresh=0.9,
                 overlap_thresh=0.9,
                 clip_model_type='ViT-B/32',
                 prompts=['cucumber', 'leaf', 'blob'],
                 ):
        self.device = device
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor_type = sam_predictor_type
        if self.sam_predictor_type == 'auto':
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)      # TODO: parameters?
        elif self.sam_predictor_type == 'points':
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=None, point_grids=[point_coords])
        else:
            raise ValueError

        # parameters for mask filtering
        self.crop = crop
        self.min_area = min_area
        self.max_area = max_area
        self.iou_thresh = iou_thresh
        self.overlap_thresh = overlap_thresh

        # CLIP model
        self.clip, self.clip_preprocess = clip.load(clip_model_type, device=self.device)
        self.prompts = prompts
        self.text_features = self.clip_prompt_encode(prompts)

    def generate_masks(self, img):
        img = self.crop_image(img)
        h, w = img.shape[:2]
        masks = self.mask_generator.generate(img)
        min_max_area_masks = []
        for m in masks:
            if self.min_area * h * w < m['area'] < self.max_area * h * w:
                m['segmentation'], m['area'], m['bbox'] = select_largest_contour(m)
                min_max_area_masks.append(m)

        # merge_indices, merge_masks = self.find_overlap_masks(min_max_area_masks)
        return min_max_area_masks

    def find_overlap_masks(self, masks: list):
        indices = set(range(len(masks)))
        merge_indices = {}
        merge_masks = {}
        while indices:
            pivot = indices.pop()
            merge_indices[pivot] = []
            merge_masks[pivot] = {'segmentation': masks[pivot]['segmentation'],
                                  'area': masks[pivot]['area'],
                                  'bbox': masks[pivot]['bbox']}
            for i in indices:
                pair_stats = calc_mask_iou(merge_masks[pivot], masks[i])
                if pair_stats['iou'] > self.iou_thresh or pair_stats['m2_overlap'] > self.overlap_thresh:
                    merge_indices[pivot].append(i)
                    merge_masks[pivot]['segmentation'] = pair_stats['union']
                    merge_masks[pivot]['area'] = pair_stats['union_area']
                    merge_masks[pivot]['bbox'] = pair_stats['union_box']
            indices.difference_update(merge_indices[pivot])

        merge_masks_list = list(merge_masks.values())
        return merge_indices, merge_masks, merge_masks_list

    def clip_prompt_encode(self, prompts: list):
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {p}") for p in prompts]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_inputs)
        return text_features

    def clip_image_encode(self, image):
        if isinstance(image, Image.Image):
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image, mode='RGB')     # RGB image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        else:
            raise ValueError
        with torch.no_grad():
            image_features = self.clip.encode_image(image_input)
        return image_features

    def mask_classification(self, masks: list, image, background_color=255):
        # cut_out_imgs = cut_out(masks, image, background_color)
        cut_out_imgs = cut_out(masks, image, pad=10)
        image_features = [self.clip_image_encode(img) for img in cut_out_imgs]
        image_features = torch.cat(image_features, dim=0)
        text_features = self.text_features.clone()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity, torch.argmax(similarity, dim=1)

    def crop_image(self, img):
        if self.crop is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = self.crop
            img = img[crop_y1: crop_y2, crop_x1: crop_x2]
        return img
