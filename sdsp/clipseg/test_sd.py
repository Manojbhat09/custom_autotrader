import torch
import requests
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from clipseg.models.clipseg import CLIPDensePredT
import cv2
from torch import autocast
from diffusers import StableDiffusionInpaintPipeline
from datetime import datetime
import os

token =  #os.environ.get('HUGGINGFACE_TOKEN')

# Define device for model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIPDensePredT model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(device)
model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=device), strict=False)
model.eval()
 
# Load Stable Diffusion Inpaint Pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=token
).to(device)

# Define transformation for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((512, 512)),
])

# Image save path
save_path = "/home/mbhat/tradebot/custom_autotrader/sdsp/clipseg/data/images"

# Helper function to save images
def save_image(image, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_path, f"{prefix}_{timestamp}.png")
    image.save(filename)
    return filename

# Load and process the input image
image_url = 'https://okmagazine.ge/wp-content/uploads/2021/04/00-promo-rob-pattison-1024x1024.jpg'
input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
img = transform(input_image).unsqueeze(0).to(device)

# Save the input image
init_image_path = save_image(input_image.resize((512, 512)), "init_image")

# Prediction prompts
prompts = ['shirt']

# Predict with the model
with torch.no_grad():
    preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

# Process and save mask for inpainting
img2 = torch.sigmoid(preds[0][0]).cpu()
transform = T.ToPILImage()
img2 = transform(img2)
mask_filename = save_image(img2, "mask")

# Process mask for inpainting
img2 = cv2.imread(mask_filename)
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
_, bw_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
mask_filename_bw = save_image(Image.fromarray(bw_image), "mask_bw")

# Inpainting with Stable Diffusion
init_image = Image.open(init_image_path)
mask = Image.open(mask_filename_bw)

with autocast(device):
    import pdb; pdb.set_trace()
    images = pipe(prompt="a blue floral holiday casual shirt", image=init_image, mask_image=mask, strength=0.8)["sample"]

# Save the inpainted image
inpaint_image_path = save_image(images[0], "inpainted")
