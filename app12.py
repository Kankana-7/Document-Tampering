
### FINAL SCRIPT

from dtd import seg_dtd
import torch
from albumentations import ToTensorV2
import torchvision
import cv2
import jpegio
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import os
import io
from io import BytesIO
from tempfile import NamedTemporaryFile

import asyncio


model = seg_dtd(n_class=2)
weights = torch.load("/home/xelpmoc/Documents/DocTamper/seg_dtd_model_weights.pth")
model.load_state_dict(weights)
model.eval()

new_qtb = (
    np.array(
        [
            [2, 1, 1, 2, 2, 4, 5, 6],
            [1, 1, 1, 2, 3, 6, 6, 6],
            [1, 1, 2, 2, 4, 6, 7, 6],
            [1, 2, 2, 3, 5, 9, 8, 6],
            [2, 2, 4, 6, 7, 11, 10, 8],
            [2, 4, 6, 6, 8, 10, 11, 9],
            [5, 6, 8, 9, 10, 12, 12, 10],
            [7, 9, 10, 10, 11, 10, 10, 10],
        ],
        dtype=np.int32,
    )
    .reshape(
        64,
    )
    .tolist()
)

totsr = ToTensorV2()
toctsr = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)
        ),
    ]
)

device=torch.device("cpu")
model=model.to(device)
print(device)


def crop_img(img, jpg_dct, crop_size=512, mask=None):
    if mask is None:
        use_mask = False
    else:
        use_mask = True
        crop_masks = []

    h, w, c = img.shape
    h_grids = h // crop_size
    w_grids = w // crop_size

    crop_imgs = []
    crop_jpe_dcts = []

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_img = img[y1:y2, x1:x2, :]
            crop_imgs.append(crop_img)
            crop_jpe_dct = jpg_dct[y1:y2, x1:x2]
            crop_jpe_dcts.append(crop_jpe_dct)
            if use_mask:
                if mask[y1:y2, x1:x2].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w % crop_size != 0:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            crop_imgs.append(img[y1:y2, w - 512 : w, :])
            crop_jpe_dcts.append(jpg_dct[y1:y2, w - 512 : w])
            if use_mask:
                if mask[y1:y2, w - 512 : w].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if h % crop_size != 0:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            crop_imgs.append(img[h - 512 : h, x1:x2, :])
            crop_jpe_dcts.append(jpg_dct[h - 512 : h, x1:x2])
            if use_mask:
                if mask[h - 512 : h, x1:x2].max() != 0:
                    crop_masks.append(1)
                else:
                    crop_masks.append(0)

    if w % crop_size != 0 and h % crop_size != 0:
        crop_imgs.append(img[h - 512 : h, w - 512 : w, :])
        crop_jpe_dcts.append(jpg_dct[h - 512 : h, w - 512 : w])
        if use_mask:
            if mask[h - 512 : h, w - 512 : w].max() != 0:
                crop_masks.append(1)
            else:
                crop_masks.append(0)

    if use_mask:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, crop_masks
    else:
        return crop_imgs, crop_jpe_dcts, h_grids, w_grids, None
    
def combine_img(imgs, h_grids, w_grids, img_h, img_w, crop_size=512):
    i = 0
    re_img = np.zeros((img_h, img_w))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, x1:x2] = imgs[i]
            i += 1

    if w_grids * crop_size < img_w:
        for h_idx in range(h_grids):
            y1 = h_idx * crop_size
            y2 = y1 + crop_size
            re_img[y1:y2, img_w - 512 : img_w] = imgs[i]
            i += 1

    if h_grids * crop_size < img_h:
        for w_idx in range(w_grids):
            x1 = w_idx * crop_size
            x2 = x1 + crop_size
            re_img[img_h - 512 : img_h, x1:x2] = imgs[i]
            i += 1

    if w_grids * crop_size < img_w and h_grids * crop_size < img_h:
        re_img[img_h - 512 : img_h, img_w - 512 : img_w] = imgs[i]

    return re_img

# # Streamlit App Title
# st.title("Image Cropper with Overlapping Patches")

# # Upload Image
# uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

# if uploaded_file is not None:
#     # Convert uploaded file into a NumPy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     imgs_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     st.text(f"Original Image Type: {type(imgs_original)}")
    
#     img = Image.fromarray(cv2.cvtColor(imgs_original, cv2.COLOR_BGR2RGB))

#     # Resize Image with Zoom Factor
#     crop_size = 512
#     zoom_factor = 1.1  # 10% zoom-in
#     new_width = int(img.width * zoom_factor)
#     new_height = int(img.height * zoom_factor)
#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Define Overlapping Cropping Strategy
#     stride = int(crop_size * 0.6)  # 60% overlap
#     width, height = img.size
#     cropped_images = {}

#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)

#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

#     for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
#         box = (x, y, x + crop_size, y + crop_size)
#         cropped_image = img.crop(box)
        
#         # Store in memory using BytesIO
#         img_io = BytesIO()
#         cropped_image.save(img_io, format="JPEG")
#         img_io.seek(0)  # Reset the stream position

#         cropped_images[f"cropped_{idx+1}.jpg"] = img_io

#     st.success(f"Stored {len(cropped_images)} cropped images in memory!")
    
#     # Initialize a list to store images for displaying
#     img_list = []
#     prediction_list = []

#     for idx, (key, value) in enumerate(cropped_images.items()):
#         # Process the cropped image for DCT coefficient extraction
#         sample_img = Image.open(value)

#         imgs_ori = np.array(sample_img)  # Convert to numpy array for DCT processing
        
#         if imgs_ori is not None:
#             # Save the image temporarily to read with jpegio
#             with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#                 temp_path = temp_file.name
#                 cv2.imwrite(temp_path, imgs_ori)

#             # Read the DCT coefficients using jpegio
#             jpg_dct = jpegio.read(temp_path)
#             dct_ori = jpg_dct.coef_arrays[0].copy()
#             use_qtb2 = jpg_dct.quant_tables[0].copy()

#             # Get image dimensions
#             h, w, c = imgs_ori.shape

#             if h % 8 == 0 and w % 8 == 0:
#                 imgs_d = imgs_ori
#                 dct_d = dct_ori
#             else:
#                 imgs_d = imgs_ori[0:(h//8)*8, 0:(w//8)*8, :].copy()
#                 dct_d = dct_ori[0:(h//8)*8, 0:(w//8)*8].copy()

#             qs = torch.LongTensor(use_qtb2)
#             img_h, img_w, _ = imgs_d.shape

#             # Assuming `crop_img` and `toctsr` functions are defined elsewhere
#             crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

#             # Process all crops and their predictions
#             for i, crop in enumerate(crop_imgs):
#                 crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#                 data = toctsr(crop)
#                 dct = torch.LongTensor(crop_jpe_dcts[i])

#                 data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
#                 dct = torch.abs(dct).clamp(0, 20)

#                 B, C, H, W = data.shape
#                 qs = qs.reshape(B, 1, 8, 8)

#                 with torch.no_grad():
#                     if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size((512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
#                         pred = model(data, dct, qs)
#                         pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
#                         prediction_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))  # Store prediction for each crop

#             # Ensure `ci` is properly formatted for display
#             ci = prediction_list[i]  # Get the prediction corresponding to the i-th crop
#             ci = ci.squeeze()  # Remove unnecessary dimensions

#             # Ensure grayscale format for display
#             if len(ci.shape) == 2:  # If already 2D, no need to convert
#                 ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel grayscale

#             # Add the cropped image to img_list
#             img_list.append(np.array(crop))

#     # Now display all the images and predictions
#     for idx, crop_img in enumerate(img_list):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Display the cropped image (ensure it's in RGB format)
#             st.image(crop_img, caption=f"Cropped Image {idx+1}", width=300)

#         with col2:
#             # Process the prediction image
#             pred_img = prediction_list[idx]

#             # Ensure pred_img is 2D (grayscale) and convert to 3-channel RGB for display
#             if len(pred_img.shape) == 2:
#                 # If already 2D, convert it to a 3-channel image
#                 pred_img = np.stack([pred_img] * 3, axis=-1)
            
#             # Ensure the shape is correct for display
#             if pred_img.shape[0] == 1:  # Single channel prediction
#                 pred_img = pred_img.squeeze(0)  # Remove the channel dimension

#             # Display the tampered region (prediction image)
#             st.image(pred_img, width=300, caption=f"Tampered Region {idx+1}")

#############################################################################################################

# # Function to handle file upload and process the image
# uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

# if uploaded_file is not None:
#     # Convert uploaded file into a NumPy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     imgs_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     st.text(f"Original Image Type: {type(imgs_original)}")

#     # Convert to PIL Image
#     img = Image.fromarray(cv2.cvtColor(imgs_original, cv2.COLOR_BGR2RGB))

#     # Resize Image with Zoom Factor
#     crop_size = 512
#     zoom_factor = 1.1  # 10% zoom-in
#     new_width = int(img.width * zoom_factor)
#     new_height = int(img.height * zoom_factor)
#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Define Overlapping Cropping Strategy
#     stride = int(crop_size * 0.6)  # 60% overlap
#     width, height = img.size
#     cropped_images = {}

#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)

#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

#     for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
#         box = (x, y, x + crop_size, y + crop_size)
#         cropped_image = img.crop(box)

#         # Store in memory using BytesIO
#         img_io = BytesIO()
#         cropped_image.save(img_io, format="JPEG")
#         img_io.seek(0)  # Reset the stream position

#         cropped_images[f"cropped_{idx+1}.jpg"] = img_io

#     st.success(f"Stored {len(cropped_images)} cropped images in memory!")

#     # Process all cropped images
#     all_results = []
#     all_extracted_texts = []
    
#     for img_key in cropped_images:
#         # Load the current cropped image
#         current_img = Image.open(cropped_images[img_key])
        
#         # Process the current_img as `imgs_ori` for DCT coefficient extraction
#         imgs_ori = np.array(current_img)  # Convert to numpy array for DCT processing

#         if imgs_ori is not None:
#             # Save the image temporarily to read with jpegio
#             with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#                 temp_path = temp_file.name
#                 cv2.imwrite(temp_path, imgs_ori)

#             # Read the DCT coefficients using jpegio
#             jpg_dct = jpegio.read(temp_path)
#             dct_ori = jpg_dct.coef_arrays[0].copy()
#             use_qtb2 = jpg_dct.quant_tables[0].copy()

#             # Get image dimensions
#             h, w, c = imgs_ori.shape

#             if h % 8 == 0 and w % 8 == 0:
#                 imgs_d = imgs_ori
#                 dct_d = dct_ori
#             else:
#                 imgs_d = imgs_ori[0:(h//8)*8, 0:(w//8)*8, :].copy()
#                 dct_d = dct_ori[0:(h//8)*8, 0:(w//8)*8].copy()

#             qs = torch.LongTensor(use_qtb2)
#             img_h, img_w, _ = imgs_d.shape

#             # Assuming `crop_img` and `toctsr` functions are defined elsewhere
#             crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

#             img_list = []
#             for idx, crop in enumerate(crop_imgs):
#                 crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#                 data = toctsr(crop)
#                 dct = torch.LongTensor(crop_jpe_dcts[idx])

#                 data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
#                 dct = torch.abs(dct).clamp(0, 20)

#                 B, C, H, W = data.shape
#                 qs = qs.reshape(B, 1, 8, 8)

#                 with torch.no_grad():
#                     if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size((512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
#                         pred = model(data, dct, qs)
#                         pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
#                         img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

#             # Process the results for this image
#             if len(img_list) > 0:
#                 ci = img_list[0]  # Take the first element from the list
#                 ci = ci.squeeze()  # Remove unnecessary dimensions

#                 # Ensure grayscale format for display
#                 if len(ci.shape) == 2:  # If already 2D, no need to convert
#                     ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel grayscale

#                 # Store the original and processed images
#                 all_results.append((imgs_ori, ci))
                
#                 # Process the mask for text extraction
#                 mask = ci  # Assuming `ci` contains the final mask
#                 mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#                 # Apply threshold to get a binary mask
#                 threshold_value = 50  # Lower to capture more region
                
#                 if len(mask.shape) == 3:
#                     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#                 _, binary_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)

#                 # Perform morphological operations
#                 kernel = np.ones((7, 7), np.uint8)
#                 binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
#                 binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)

#                 # Find contours
#                 contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                 # Text extraction for this image
#                 extracted_texts = {}
#                 min_area = 500

#                 if len(contours) > 0:
#                     for contour in contours:
#                         if cv2.contourArea(contour) > min_area:
#                             hull = cv2.convexHull(contour)
#                             x, y, w, h = cv2.boundingRect(hull)

#                             # Crop and process for OCR
#                             gray_img = cv2.cvtColor(imgs_ori, cv2.COLOR_BGR2GRAY)
#                             cropped_region = gray_img[y:y + h, x:x + w]
#                             cropped_region = cv2.GaussianBlur(cropped_region, (3,3), 0)
#                             _, cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                             extracted_text = pytesseract.image_to_string(cropped_region, config="--psm 6")
#                             extracted_texts[(x, y, w, h)] = extracted_text.strip()

#                 all_extracted_texts.append((img_key, extracted_texts))

#     # Display results for all images
#     st.subheader("Processing Results for All Cropped Images")
    
#     # Display in a grid format
#     cols_per_row = 4
#     rows = (len(all_results) + cols_per_row - 1) // cols_per_row
    
#     for row in range(rows):
#         cols = st.columns(cols_per_row)
#         for col_idx in range(cols_per_row):
#             img_idx = row * cols_per_row + col_idx
#             if img_idx < len(all_results):
#                 with cols[col_idx]:
#                     imgs_ori, ci = all_results[img_idx]
#                     img_key = list(cropped_images.keys())[img_idx]
                    
#                     # Display original and processed images
#                     st.image(imgs_ori, caption=f"Original {img_key}", use_column_width=True)
#                     st.image(ci, caption=f"Tampered Regions {img_key}", use_column_width=True)
                    
#                     # Display extracted text if available
#                     if img_idx < len(all_extracted_texts):
#                         img_key, extracted_texts = all_extracted_texts[img_idx]
#                         if extracted_texts:
#                             st.text(f"Extracted text from {img_key}:")
#                             for (x, y, w, h), text in extracted_texts.items():
#                                 st.text(f"Region ({x},{y})-({x+w},{y+h}): {text}")

########################################################################################################################

# # Function to handle file upload and process the image
# uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

# if uploaded_file is not None:
#     # Convert uploaded file into a NumPy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     imgs_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     st.text(f"Original Image Type: {type(imgs_original)}")

#     # Convert to PIL Image
#     img = Image.fromarray(cv2.cvtColor(imgs_original, cv2.COLOR_BGR2RGB))

#     # Resize Image with Zoom Factor
#     crop_size = 512
#     zoom_factor = 1.1  # 10% zoom-in
#     new_width = int(img.width * zoom_factor)
#     new_height = int(img.height * zoom_factor)
#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Define Overlapping Cropping Strategy
#     stride = int(crop_size * 0.6)  # 60% overlap
#     width, height = img.size
#     cropped_images = {}

#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)

#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

#     for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
#         box = (x, y, x + crop_size, y + crop_size)
#         cropped_image = img.crop(box)

#         # Store in memory using BytesIO
#         img_io = BytesIO()
#         cropped_image.save(img_io, format="JPEG")
#         img_io.seek(0)  # Reset the stream position

#         cropped_images[f"cropped_{idx+1}.jpg"] = img_io

#     st.success(f"Stored {len(cropped_images)} cropped images in memory!")

#     # Process all cropped images
#     for img_key in cropped_images:
#         st.subheader(f"Processing {img_key}")
        
#         # Load the current cropped image
#         current_img = Image.open(cropped_images[img_key])
        
#         # Convert to numpy array for processing
#         imgs_ori = np.array(current_img)
        
#         # Save the image temporarily to read with jpegio
#         with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#             temp_path = temp_file.name
#             cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

#         # Read the DCT coefficients using jpegio
#         jpg_dct = jpegio.read(temp_path)
#         dct_ori = jpg_dct.coef_arrays[0].copy()
#         use_qtb2 = jpg_dct.quant_tables[0].copy()

#         # Get image dimensions
#         h, w, c = imgs_ori.shape

#         if h % 8 == 0 and w % 8 == 0:
#             imgs_d = imgs_ori
#             dct_d = dct_ori
#         else:
#             imgs_d = imgs_ori[0:(h//8)*8, 0:(w//8)*8, :].copy()
#             dct_d = dct_ori[0:(h//8)*8, 0:(w//8)*8].copy()

#         qs = torch.LongTensor(use_qtb2)
#         img_h, img_w, _ = imgs_d.shape

#         # Crop the image and process DCT coefficients
#         crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

#         img_list = []
#         for idx, crop in enumerate(crop_imgs):
#             crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             data = toctsr(crop)
#             dct = torch.LongTensor(crop_jpe_dcts[idx])

#             data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
#             dct = torch.abs(dct).clamp(0, 20)

#             B, C, H, W = data.shape
#             qs = qs.reshape(B, 1, 8, 8)

#             with torch.no_grad():
#                 if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size((512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
#                     pred = model(data, dct, qs)
#                     pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
#                     img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

#         # Process the results for this image
#         if len(img_list) > 0:
#             ci = img_list[0]  # Take the first element from the list
#             ci = ci.squeeze()  # Remove unnecessary dimensions

#             # Ensure grayscale format for display
#             if len(ci.shape) == 2:  # If already 2D, no need to convert
#                 ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel grayscale

#             # Create the four output images
#             # 1. Original cropped image
#             original_img = imgs_ori.copy()
            
#             # 2. Predicted mask
#             mask = ci.copy()
            
#             # Process the mask for bounding boxes
#             mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
#             _, binary_mask = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
            
#             # Morphological operations
#             kernel = np.ones((7, 7), np.uint8)
#             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
#             binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            
#             # Find contours
#             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # 3. Predicted mask with bounding boxes
#             mask_with_boxes = mask.copy()
            
#             # 4. Original image with bounding boxes
#             original_with_boxes = original_img.copy()
            
#             # Text extraction
#             extracted_texts = {}
#             min_area = 500
            
#             if len(contours) > 0:
#                 for contour in contours:
#                     if cv2.contourArea(contour) > min_area:
#                         hull = cv2.convexHull(contour)
#                         x, y, w, h = cv2.boundingRect(hull)
                        
#                         # Draw bounding boxes
#                         cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.rectangle(original_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
#                         # Extract text from the original image
#                         gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
#                         cropped_region = gray_img[y:y + h, x:x + w]
#                         cropped_region = cv2.GaussianBlur(cropped_region, (3, 3), 0)
#                         _, cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                         extracted_text = pytesseract.image_to_string(cropped_region, config="--psm 6")
#                         extracted_texts[(x, y, w, h)] = extracted_text.strip()

#             # Display all four images in a 2x2 grid
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(original_img, caption="1. Original Cropped Image", width = 300)
#                 st.image(mask_with_boxes, caption="3. Predicted Mask with Bounding Boxes", width = 300)
#             with col2:
#                 st.image(mask, caption="2. Predicted Mask", width = 300)
#                 st.image(original_with_boxes, caption="4. Original with Bounding Boxes", width = 300)
            
#             # Display extracted text
#             if extracted_texts:
#                 st.subheader("Extracted Text from Tampered Regions")
#                 for (x, y, w, h), text in extracted_texts.items():
#                     st.text(f"Region ({x}, {y}, {w}, {h}): {text}")
#             else:
#                 st.text("No text extracted from this image")
            
#             st.markdown("---")  # Add separator between images

########################################################################################################################

# # Function to handle file upload and process the image
# uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

# if uploaded_file is not None:
#     # Convert uploaded file into a NumPy array
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     imgs_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     st.text(f"Original Image Type: {type(imgs_original)}")

#     # Convert to PIL Image
#     img = Image.fromarray(cv2.cvtColor(imgs_original, cv2.COLOR_BGR2RGB))

#     # Resize Image with Zoom Factor
#     crop_size = 512
#     zoom_factor = 1.1  # 10% zoom-in
#     new_width = int(img.width * zoom_factor)
#     new_height = int(img.height * zoom_factor)
#     img = img.resize((new_width, new_height), Image.LANCZOS)

#     # Define Overlapping Cropping Strategy
#     stride = int(crop_size * 0.6)  # 60% overlap
#     width, height = img.size
#     cropped_images = {}

#     y_positions = list(range(0, height - crop_size + 1, stride))
#     if (height - crop_size) % stride != 0:
#         y_positions.append(height - crop_size)

#     x_positions = list(range(0, width - crop_size + 1, stride))
#     if (width - crop_size) % stride != 0:
#         x_positions.append(width - crop_size)

#     for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
#         box = (x, y, x + crop_size, y + crop_size)
#         cropped_image = img.crop(box)

#         # Store in memory using BytesIO
#         img_io = BytesIO()
#         cropped_image.save(img_io, format="JPEG")
#         img_io.seek(0)  # Reset the stream position

#         cropped_images[f"cropped_{idx+1}.jpg"] = img_io

#     st.success(f"Stored {len(cropped_images)} cropped images in memory!")

#     # Process all cropped images
#     for img_idx, img_key in enumerate(cropped_images):
#         st.subheader(f"Cropped Image {img_idx + 1}")
        
#         # Load the current cropped image
#         current_img = Image.open(cropped_images[img_key])
        
#         # Convert to numpy array for processing
#         imgs_ori = np.array(current_img)
        
#         # Save the image temporarily to read with jpegio
#         with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
#             temp_path = temp_file.name
#             cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

#         # Read the DCT coefficients using jpegio
#         jpg_dct = jpegio.read(temp_path)
#         dct_ori = jpg_dct.coef_arrays[0].copy()
#         use_qtb2 = jpg_dct.quant_tables[0].copy()

#         # Get image dimensions
#         h, w, c = imgs_ori.shape

#         if h % 8 == 0 and w % 8 == 0:
#             imgs_d = imgs_ori
#             dct_d = dct_ori
#         else:
#             imgs_d = imgs_ori[0:(h//8)*8, 0:(w//8)*8, :].copy()
#             dct_d = dct_ori[0:(h//8)*8, 0:(w//8)*8].copy()

#         qs = torch.LongTensor(use_qtb2)
#         img_h, img_w, _ = imgs_d.shape

#         # Crop the image and process DCT coefficients
#         crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

#         img_list = []
#         for idx, crop in enumerate(crop_imgs):
#             crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#             data = toctsr(crop)
#             dct = torch.LongTensor(crop_jpe_dcts[idx])

#             data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
#             dct = torch.abs(dct).clamp(0, 20)

#             B, C, H, W = data.shape
#             qs = qs.reshape(B, 1, 8, 8)

#             with torch.no_grad():
#                 if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size((512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
#                     pred = model(data, dct, qs)
#                     pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
#                     img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

#         # Process the results for this image
#         if len(img_list) > 0:
#             ci = img_list[0]  # Take the first element from the list
#             ci = ci.squeeze()  # Remove unnecessary dimensions

#             # Ensure grayscale format for display
#             if len(ci.shape) == 2:  # If already 2D, no need to convert
#                 ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel grayscale

#             # Create the four output images
#             # 1. Original cropped image
#             original_img = imgs_ori.copy()
            
#             # 2. Predicted mask
#             mask = ci.copy()
            
#             # Process the mask for bounding boxes
#             mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
#             _, binary_mask = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
            
#             # Morphological operations
#             kernel = np.ones((7, 7), np.uint8)
#             binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
#             binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            
#             # Find contours
#             contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # 3. Predicted mask with bounding boxes
#             mask_with_boxes = mask.copy()
            
#             # 4. Original image with bounding boxes
#             original_with_boxes = original_img.copy()
            
#             # Text extraction
#             extracted_texts = {}
#             min_area = 500
            
#             if len(contours) > 0:
#                 for contour in contours:
#                     if cv2.contourArea(contour) > min_area:
#                         hull = cv2.convexHull(contour)
#                         x, y, w, h = cv2.boundingRect(hull)
                        
#                         # Draw bounding boxes
#                         cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                         cv2.rectangle(original_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
#                         # Extract text from the original image
#                         gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
#                         cropped_region = gray_img[y:y + h, x:x + w]
#                         cropped_region = cv2.GaussianBlur(cropped_region, (3, 3), 0)
#                         _, cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                         extracted_text = pytesseract.image_to_string(cropped_region, config="--psm 6")
#                         extracted_texts[(x, y, w, h)] = extracted_text.strip()

#             # Display all four images in a 2x2 grid with dynamic captions
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(original_img, caption=f"Cropped Image {img_idx+1}_1", width = 300)
#                 st.image(mask_with_boxes, caption=f"Cropped Image {img_idx+1}_3", width = 300)
#             with col2:
#                 st.image(mask, caption=f"Cropped Image {img_idx+1}_2", width = 300)
#                 st.image(original_with_boxes, caption=f"Cropped Image {img_idx+1}_4", width = 300)
            
#             # Display extracted text
#             if extracted_texts:
#                 st.subheader("Extracted Text from Tampered Regions")
#                 for (x, y, w, h), text in extracted_texts.items():
#                     st.text(f"Region ({x}, {y}, {w}, {h}): {text}")
#             else:
#                 st.text("No text extracted from this image")
            
#             st.markdown("---")  # Add separator between images

st.title("Document Tampering Detection System")
# Function to handle file upload and process the image
uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file into a NumPy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgs_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(imgs_original, caption="Uploaded Image", use_container_width=True,channels="BGR")

    # Convert to PIL Image
    img = Image.fromarray(cv2.cvtColor(imgs_original, cv2.COLOR_BGR2RGB))

    # Resize Image with Zoom Factor
    crop_size = 512
    zoom_factor = 1.1  # 10% zoom-in
    new_width = int(img.width * zoom_factor)
    new_height = int(img.height * zoom_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Define Overlapping Cropping Strategy
    stride = int(crop_size * 0.7)  # 70% overlap
    width, height = img.size
    cropped_images = {}

    y_positions = list(range(0, height - crop_size + 1, stride))
    if (height - crop_size) % stride != 0:
        y_positions.append(height - crop_size)

    x_positions = list(range(0, width - crop_size + 1, stride))
    if (width - crop_size) % stride != 0:
        x_positions.append(width - crop_size)

    for idx, (y, x) in enumerate([(y, x) for y in y_positions for x in x_positions]):
        box = (x, y, x + crop_size, y + crop_size)
        cropped_image = img.crop(box)

        # Store in memory using BytesIO
        img_io = BytesIO()
        cropped_image.save(img_io, format="JPEG")
        img_io.seek(0)  # Reset the stream position

        # Calculate the main image number and sub-image number
        main_img_num = (idx // len(x_positions)) + 1
        sub_img_num = (idx % len(x_positions)) + 1
        cropped_images[f"Cropped Image {main_img_num}_{sub_img_num}.jpg"] = img_io

    #st.success(f"Stored {len(cropped_images)} cropped images in memory!")

    # Process all cropped images
    for img_key in cropped_images:
        st.subheader(f"Processing {img_key}")
        
        # Load the current cropped image
        current_img = Image.open(cropped_images[img_key])
        
        # Convert to numpy array for processing
        imgs_ori = np.array(current_img)
        
        # Save the image temporarily to read with jpegio
        with NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, cv2.cvtColor(imgs_ori, cv2.COLOR_RGB2BGR))

        # Read the DCT coefficients using jpegio
        jpg_dct = jpegio.read(temp_path)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        # Get image dimensions
        h, w, c = imgs_ori.shape

        if h % 8 == 0 and w % 8 == 0:
            imgs_d = imgs_ori
            dct_d = dct_ori
        else:
            imgs_d = imgs_ori[0:(h//8)*8, 0:(w//8)*8, :].copy()
            dct_d = dct_ori[0:(h//8)*8, 0:(w//8)*8].copy()

        qs = torch.LongTensor(use_qtb2)
        img_h, img_w, _ = imgs_d.shape

        # Crop the image and process DCT coefficients
        crop_imgs, crop_jpe_dcts, h_grids, w_grids, _ = crop_img(imgs_d, dct_d, crop_size=512, mask=None)

        img_list = []
        for idx, crop in enumerate(crop_imgs):
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            data = toctsr(crop)
            dct = torch.LongTensor(crop_jpe_dcts[idx])

            data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
            dct = torch.abs(dct).clamp(0, 20)

            B, C, H, W = data.shape
            qs = qs.reshape(B, 1, 8, 8)

            with torch.no_grad():
                if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size((512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
                    pred = model(data, dct, qs)
                    pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
                    img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))

        # Process the results for this image
        if len(img_list) > 0:
            ci = img_list[0]  # Take the first element from the list
            ci = ci.squeeze()  # Remove unnecessary dimensions

            # Ensure grayscale format for display
            if len(ci.shape) == 2:  # If already 2D, no need to convert
                ci = np.stack([ci] * 3, axis=-1)  # Convert to 3-channel grayscale

            # Create the four output images
            # 1. Original cropped image
            original_img = imgs_ori.copy()
            
            # 2. Predicted mask
            mask = ci.copy()
            # Process the mask for bounding boxes
            threshold_value = 50
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) if len(mask.shape) == 3 else mask
            _, binary_mask = cv2.threshold(mask_gray, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = np.ones((7, 7), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. Predicted mask with bounding boxes
            mask_with_boxes = mask.copy()
            
            # 4. Original image with bounding boxes
            original_with_boxes = original_img.copy()
            
            # Text extraction
            extracted_texts = {}
            min_area = 500
            
            if len(contours) > 0:
                for contour in contours:
                    if cv2.contourArea(contour) > min_area:
                        hull = cv2.convexHull(contour)
                        x, y, w, h = cv2.boundingRect(hull)
                        
                        # Draw bounding boxes
                        cv2.rectangle(mask_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.rectangle(original_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Extract text from the original image
                        gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
                        cropped_region = gray_img[y:y + h, x:x + w]
                        cropped_region = cv2.GaussianBlur(cropped_region, (3, 3), 0)
                        _, cropped_region = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        extracted_text = pytesseract.image_to_string(cropped_region, config="--psm 6")
                        extracted_texts[(x, y, w, h)] = extracted_text.strip()

            # Display all four images in a 2x2 grid
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption=f"1. Original", width=300)
                st.image(mask_with_boxes, caption=f"3. Predicted Mask with Bounding Boxes", width=300)
            with col2:
                st.image(mask, caption=f"2. Predicted Mask", width=300)
                st.image(original_with_boxes, caption=f"4. Original with Bounding Boxes", width=300)
            
            # Display extracted text
            if extracted_texts:
                st.markdown("<h4>Extracted Text from Tampered Regions</h4>", unsafe_allow_html=True)
                for (x, y, w, h), text in extracted_texts.items():
                    st.text(f"Region ({x}, {y}, {w}, {h}): {text}")
            else:
                st.text(f"No text extracted...")
            
            st.markdown("---")  # Add separator between images