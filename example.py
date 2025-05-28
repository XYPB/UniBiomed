import argparse
import torch
from transformers import (AutoModel, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          GenerationConfig)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='UniBiomed')
    parser.add_argument('--model_path', default='Luffy503/UniBiomed')
    args = parser.parse_args()
    return args


def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([30, 255, 30])
    cmap[255] = np.array([0, 0, 0])

    return cmap


def blend(img, lab):
    img = img.convert('RGB')
    lab = lab.convert('RGB')
    ble = Image.blend(img, lab, 0.6)
    return ble


def blend_final(img, lab):
    ble = blend(img, lab)
    ble = np.asarray(ble)

    img = img.convert('RGB')
    lab = lab.convert('RGB')

    array1 = np.array(img)
    array2 = np.array(lab)

    mask = array2 == 0 

    blended_array = np.where(mask, array1, ble)

    blended_image = Image.fromarray(blended_array.astype(np.uint8))

    return blended_image


args = parse_args()


# load model
model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
)

# define data input, image and text instruction
data_dict = {}

# Segmentation ------------------------------------------------------------
image_file = './examples/pathology_multiple.png'
image = Image.open(image_file).convert('RGB')
text = f"<image>Please segment nuclei in pathology."
data_dict['image'] = image
data_dict['text'] = text
pred_dict = model.predict_forward(**data_dict, tokenizer=tokenizer)
# text description
prediction = pred_dict['prediction']
# segmentation mask
mask = pred_dict['prediction_masks'][0][0]
print(prediction)

# visualize segmentation
cmap = color_map()
mask = Image.fromarray(mask.astype(np.uint8), mode='P')
mask.putpalette(cmap)
blend_output = blend_final(image, mask)

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(image)
axs[0].axis("off")
axs[1].imshow(blend_output)
axs[1].axis("off")
plt.tight_layout()
plt.show()
plt.close()


# Disease recognition ---------------------------------------------------------------
image_file = './examples/CT_lung.png'
image = Image.open(image_file).convert('RGB')
text = f"<image>Can we observe any signs of abnormality? Please respond with interleaved segmentation masks for the corresponding parts."
data_dict['image'] = image
data_dict['text'] = text
pred_dict = model.predict_forward(**data_dict, tokenizer=tokenizer)
# text description
prediction = pred_dict['prediction']
# segmentation mask
mask = pred_dict['prediction_masks'][0][0]
print(prediction)

# visualize segmentation
cmap = color_map()
mask = Image.fromarray(mask.astype(np.uint8), mode='P')
mask.putpalette(cmap)
blend_output = blend_final(image, mask)

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(image)
axs[0].axis("off")
axs[1].imshow(blend_output)
axs[1].axis("off")
plt.tight_layout()
plt.show()
plt.close()


# Report Generation ---------------------------------------------------------------
image_file = './examples/CT_abdomen.png'
image = Image.open(image_file).convert('RGB')
text = f"<image>Could you describe the abnormality in this image? Please highlight the region with interleaved segmentation masks."
data_dict['image'] = image
data_dict['text'] = text
pred_dict = model.predict_forward(**data_dict, tokenizer=tokenizer)
# text description
prediction = pred_dict['prediction']
# segmentation mask
mask = pred_dict['prediction_masks'][0][0]
print(prediction)

# visualize segmentation
cmap = color_map()
mask = Image.fromarray(mask.astype(np.uint8), mode='P')
mask.putpalette(cmap)
blend_output = blend_final(image, mask)

fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(image)
axs[0].axis("off")
axs[1].imshow(blend_output)
axs[1].axis("off")
plt.tight_layout()
plt.show()
plt.close()