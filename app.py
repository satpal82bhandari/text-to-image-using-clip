
import sys
sys.path.append('/home/adagen/Download/anaconda3/envs/python-3.8/lib/python3.8/site-packages')
import clip
import torch
from PIL import Image
import os

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory containing images
image_dir = "./images"

# List of image paths
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('jpg', 'png', 'jpeg'))]

# Preprocess and load images
images = [preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths]
images = torch.cat(images).to(device)

# Define the text query
text_query = "give me image of vector addition"
text_inputs = clip.tokenize([text_query]).to(device)

# Compute features and energy are related
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features = model.encode_text(text_inputs)

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity
similarity = (text_features @ image_features.T).squeeze(0)

# Get relevance scores and rank images
relevance_scores, indices = similarity.sort(descending=True)

# Print results
for score, idx in zip(relevance_scores, indices):
    print(f"Image: {image_paths[idx]}, Score: {score.item()}")
