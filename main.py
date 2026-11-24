import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np #added
import torchvision #addded
import torch.nn.functional as F #added
import random #added
import segmentation_models_pytorch as smp  # Allowed open-source tool

from data import train_loader, val_loader, transform

def calculate_miou(preds, masks, num_classes=3):
    """
    Calculates the mean Intersection over Union (mIoU) across 3 classes.
    """
    iou_list = []
    # Flatten the tensors to compute pixel-by-pixel matches easily
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)
    
    for cls in range(num_classes):

        intersection = ((preds_flat == cls) & (masks_flat == cls)).sum().item()
        union = ((preds_flat == cls) | (masks_flat == cls)).sum().item()
        
        if union > 0:
            iou_list.append(intersection / union)
            
    # Return the average IoU across all active classes
    return np.sum(iou_list) / num_classes if len(iou_list) > 0 else 0.0


class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # TODO: define encoder (contracting path)
        base = 64

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(base, base * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 2, base * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(base * 4, base * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True)
        )

        # TODO: define bottleneck (e.g., 512 -> 1024)
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),

            nn.Conv2d(base * 8, base * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 16, base * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3)
        )

        # TODO: define decoder (expanding path)
        self.up1 = nn.ConvTranspose2d(
            base * 16,
            base * 8,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 16, base * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 8, base * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 8, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2
        )

        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base * 2, base * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.ConvTranspose2d(
            base * 2,
            base,
            kernel_size=2,
            stride=2
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(base * 2, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),

            nn.Conv2d(base, base, kernel_size=3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        # TODO: define final 1x1 conv (out_channels = num_classes)
        self.outc = nn.Conv2d(
            base,
            num_classes,
            kernel_size=1
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encoder forward, store feature maps for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # TODO: bottleneck forward
        x5 = self.down4(x4)

        # TODO: decoder forward, concat with skip connections
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.dec3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.dec4(x)

        
        # TODO: return segmentation logits via final 1x1 conv

        logits = self.outc(x)
        return logits
    


class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    def __init__(self):
        super().__init__()
        # TODO: define the loss to use
        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss

        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, 1.5, 3.5], dtype=torch.float32)
        )

        self.ce = nn.CrossEntropyLoss(weight=self.class_weights)


    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        loss_ce = self.ce(logits, targets)

        num_classes = 3
        epsilon = 1e-6
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        loss_dice = 0.0
        # Give Class 2 higher importance in Dice calculation
        dice_weights = [1.0, 1.0, 2.0]
        
        for k in range(num_classes):
            p_k = probs[:, k, :, :]
            y_k = targets_one_hot[:, k, :, :]
            
            intersection = torch.sum(p_k * y_k, dim=(1, 2))
            union = torch.sum(p_k, dim=(1, 2)) + torch.sum(y_k, dim=(1, 2))
            
            dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)
            loss_dice += dice_weights[k] * (1.0 - dice_coeff.mean())
            
        loss_dice = loss_dice / sum(dice_weights)

        return loss_ce + (1.5 * loss_dice)


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def train_one_epoch(self, loader) -> float:
        self.model.train()
        # TODO: iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss
        total_loss = 0.0

        for images,masks in loader:
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            # Change dims to [-1] so both images and masks flip horizontally
            if np.random.rand() > 0.5:
                images = torch.flip(images, dims=[-1])
                masks = torch.flip(masks, dims=[-1])

            # Explicit dim passed to squeeze safeguarding batch size drops
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-5, 5)
                images = torchvision.transforms.functional.rotate(images, angle)
                masks = torchvision.transforms.functional.rotate(
                    masks.unsqueeze(1).float(),
                    angle,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST
                ).squeeze(dim=1).long()

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss/len(loader)


    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        # TODO: run forward only and compute loss / metrics (IoU, Dice, etc.)

        total_loss = 0.0

        for images, masks in loader:

            images = images.to(self.device)
            masks = masks.to(self.device).long()


            outputs = self.model(images)

            loss = self.criterion(outputs, masks)

            total_loss += loss.item()

        return total_loss / len(loader)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 45 # changed from 10 to 25 to 45
    learning_rate = 1e-4
    num_classes = 3

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3,classes=num_classes).to(device)
    criterion = SegmentationLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    # added to handle the increased epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    trainer = Trainer(model, criterion, optimizer, device)

    best_miou = 0.0

    images, masks = next(iter(train_loader))

    print(images.shape)
    print(masks.shape)


    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader)

        val_loss = trainer.validate(val_loader)

        # Switch model to evaluation mode gets rid of dropout and batchnorm updates
        model.eval() 
        all_preds = [] 
        all_masks = [] 
        
        # Disable gradient computation for faster inference and lower memory usage
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_masks.append(msks.cpu())
                

        # Concatenate all batch predictions into full validation set tensors
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        val_mIoU = calculate_miou(all_preds, all_masks, num_classes=3)

        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mIoU={val_mIoU:.4f}")

    

    # TODO: save best model checkpoint / visualize predictions / report metrics

        # Removed the hard threshold value check constraint blocking model improvements below 0.8
        if val_mIoU > best_miou:
            best_miou = val_mIoU
            torch.save(model.state_dict(), "best_model.pth")
            print(f"--> Best model saved! Epoch {epoch + 1} mIoU: {val_mIoU:.4f}")


        scheduler.step()


    print("training completed")


if __name__ == "__main__":
    main()