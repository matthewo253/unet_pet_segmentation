import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # Set to None as we use local trained file
    in_channels=3,
    classes=3
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval()

pred_dir = Path("test_predictions")
pred_dir.mkdir(exist_ok=True)
test_dir = Path("test_images/test_images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images = list(test_dir.glob("*.jpg"))
print("Found images:", len(images))

for img_path in images:
    image = Image.open(img_path).convert("RGB")
    original_size = image.size  # (W, H)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)  # Evaluated at 224x224

        

        # This interpolates smooth mathematical curves rather than hard integers.
        output_upscaled = F.interpolate(
            output,
            size=(original_size[1], original_size[0]),  # (Height, Width)
            mode="bilinear",
            align_corners=False
        )

        # Run argmax on the final high-resolution grid to keep boundary lines sharp
        pred = torch.argmax(output_upscaled, dim=1)
        
        # Pull to CPU, convert to NumPy, and unpack batch dimension safely
        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)

    # save prediction array 
    np.save(pred_dir / f"{img_path.stem}.npy", pred_np)
    print(img_path.stem, np.unique(pred_np))
