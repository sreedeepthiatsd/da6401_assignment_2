"""Training entrypoint
"""
"""Training entrypoint
"""
print("File is running....")
import numpy as np
import torch
import wandb
#from models.classification import VGG11Classifier
#from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
import os
os.makedirs("checkpoints", exist_ok=True)
from losses import IoULoss


def dice_score(pred, target, eps=1e-6):
    pred = torch.argmax(pred, dim=1)

    pred = (pred == 1).float()
    target = (target == 1).float()

    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))

    dice = (2 * intersection + eps) / (union + eps)

    return dice.mean()

def train(model, train_loader, val_loader, epochs=10, lr=1e-4, dropout_p=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):

    model.to(device)

    # we use this loss function as this combines softmax+log+nll
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(device))
    
    #reg_loss_fn = torch.nn.MSELoss()
    #iou_loss_fn = IoULoss()

    # we are using adam optimizer for faster convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    wandb.init(project="da6401_assignment2", name=f"unet+crossentropyloss_combined_dropout_{dropout_p}_with_bn",reinit=True)
    wandb.config.update({
    "epochs": epochs,
    "lr": lr,
    "batch_size": 32,
    "dropout": dropout_p
    })
    print("Using device: ", device)
    #best_acc = 0
    best_loss = float("inf") 
    for epoch in range(epochs):

        print(f"Epoch {epoch+1}/{epochs}")

        # ===== training ==========
        model.train()
        train_loss = 0
        train_dice_total = 0
    
        for batch in train_loader:
            images = batch["image"].to(device)
            #targets = batch["bbox"].to(device)
            #labels = batch["label"].to(device)
            targets = batch["mask"].to(device).long()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, targets) 
            dice = dice_score(outputs, targets)
            #reg_loss = reg_loss_fn(outputs, targets)
            #iou_loss = iou_loss_fn(outputs, targets)

            #loss = reg_loss + 2 * iou_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice_total += dice.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice_total / len(train_loader)

        # =========== Validation =============
        model.eval()
        val_loss = 0
        #correct = 0
        #total = 0
        val_dice_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["mask"].to(device).long()
                #targets = batch["bbox"].to(device)
                #labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                dice = dice_score(outputs, targets)
                #reg_loss = reg_loss_fn(outputs, targets)
                #iou_loss = iou_loss_fn(outputs, targets)

                #loss = reg_loss + 2 * iou_loss

                val_loss += loss.item()
                val_dice_total += dice.item()

                
        if len(val_loader) == 0:
          print("Warning: Validation loader is empty. Skipping validation.")
          avg_val_loss = 0
          avg_val_dice = 0
        else:
          avg_val_loss = val_loss / len(val_loader)
          avg_val_dice = val_dice_total / len(val_loader)
        
        #val_acc = correct / total if total > 0 else 0

        # ======== Logging ==========
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "dice_score": avg_val_dice
            #"dice_score": dice.item()
            #"reg_loss": reg_loss.item(),
            #"iou_loss": iou_loss.item()
            #"val_accuracy": val_acc,
            #"overfit_gap": avg_val_loss - avg_train_loss
        })
        torch.save(
        model.state_dict(),
        f"checkpoints/unet_epoch_{epoch+1}.pth"
        )
        # =========== save the best model =============
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss

            torch.save({
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_loss,
            },"checkpoints/unet.pth")

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f},dice_score = {avg_val_dice:.4f}")
    wandb.finish()


if __name__ == "__main__":

    print("Starting training...") 

    from data.pets_dataset import OxfordIIITPetDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # -------- TRANSFORMS --------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    from torch.utils.data import Subset
    # =========== dataset =================
    full_dataset = OxfordIIITPetDataset(
        images_dir="/content/data/images",
        annotations_file="/content/data/annotations/trainval.txt",
        xml_dir="/content/data/annotations/xmls", 
        transform=None
    )
    # ====== test dataset =======
    #sample = full_dataset[0]
    #print(sample["bbox"])
    from torch.utils.data import random_split
    # split indices
    indices = torch.randperm(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    val_size = len(full_dataset) - train_size

    train_dataset = Subset(
    OxfordIIITPetDataset(
        images_dir="/content/data/images",
        annotations_file="/content/data/annotations/trainval.txt",
        xml_dir="/content/data/annotations/xmls",
        transform=train_transform
    ),
    train_indices
    )

    val_dataset = Subset(
        OxfordIIITPetDataset(
        images_dir="/content/data/images",
        annotations_file="/content/data/annotations/trainval.txt",
        xml_dir="/content/data/annotations/xmls",
        transform=val_transform
    ),
    val_indices
    )
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    train_count = min(300, len(train_dataset))
    val_count = min(80, len(val_dataset))

    train_subset_indices = torch.randperm(len(train_dataset))[:train_count]
    train_dataset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    val_subset_indices = torch.randperm(len(val_dataset))[:val_count]
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset_indices)

    print("Reduced Train size:", len(train_dataset))
    print("Reduced Val size:", len(val_dataset))

    # val_dataset = OxfordIIITPetDataset(
    #     images_dir="/content/data/images",
    #     annotations_file="/content/data/annotations/test.txt",
    #     xml_dir="/content/data/annotations/xmls", 
    #     transform=val_transform
    #)
    
    # ========== dataloaders ====================
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

    # ======== model ==========
    dropout_p = 0.5
    model = VGG11UNet(num_classes = 2)
    # load pretrained encoder
    checkpoint = torch.load("checkpoints/classifier.pth", map_location=torch.device('cpu'))
    model.encoder.load_state_dict(checkpoint["state_dict"], strict=False)
     # freeze encoder
    # for param in model.encoder.parameters():
    #   param.requires_grad = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    targets = batch["mask"].to(device)
    #targets = batch["bbox"].to(device)

    outputs = model(images)

    print("Pred:", outputs[0])
    print("Target:", targets[0])
    print("Dataset size:", len(train_dataset))

    # =========== train the model============
    train(
        model,
        train_loader,
        val_loader,
        epochs=3,
        lr=5e-5,
        dropout_p=dropout_p,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # === visualizing bbox =========
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches

    # model.eval()

    # sample = next(iter(val_loader))
    # images = sample["image"].to(device)
    # #targets = sample["bbox"].to(device)
    # targets = sample["mask"]

    # outputs = model(images)

    # image = images[0].cpu()
    # target = targets[0].cpu()
    # pred = outputs[0].detach().cpu()

    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])

    # img = image.permute(1, 2, 0)
    # img = img * std + mean   # denormalize
    # img = img.numpy()
    # img = img.clip(0, 1)

    # fig, ax = plt.subplots(1)
    # ax.imshow(img)

    # def draw_box(box, color):
    #   x, y, w, h = box
    #   x = x.item() * 224
    #   y = y.item() * 224
    #   w = w.item() * 224
    #   h = h.item() * 224

    #   rect = patches.Rectangle((x - w/2, y - h/2),w, h,
    #     linewidth=2, edgecolor=color, facecolor='none'
    #     )
    #   ax.add_patch(rect)

    # draw_box(target, 'green')  # Ground Truth
    # draw_box(pred, 'red')      # Prediction

    # plt.title("Green = GT, Red = Prediction")
    # plt.show()

    # =================================================
    import matplotlib.pyplot as plt

    model.eval()

    sample = next(iter(val_loader))

    image = sample["image"][0]
    mask = sample["mask"][0]

    with torch.no_grad():
      pred = model(sample["image"].to(device))[0]
      pred = torch.argmax(pred, dim=0).cpu()

    # denormalize
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    img = image.permute(1,2,0)
    img = img * std + mean
    img = img.numpy().clip(0,1)

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("GT Mask")
    plt.imshow(mask)

    plt.subplot(1,3,3)
    plt.title("Pred Mask")
    plt.imshow(pred)

    plt.show()
   
