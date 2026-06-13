import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction.dataset import SOCOFingSiameseDataset
from src.feature_extraction.embedding import SiameseNetwork, ContrastiveLoss


def main():
    parser = argparse.ArgumentParser(description="Train Siamese CNN on SOCOFing")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "SOCOFing"), help="Path to SOCOFing dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_pairs", type=int, default=500, help="Number of pairs per epoch (keep small for CPU testing)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=2.0, help="Contrastive loss margin")
    parser.add_argument("--save_path", type=str, default=str(PROJECT_ROOT / "models" / "siamese_model.pth"), help="Path to save model")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard ImageNet transforms for ResNet backbone
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Initializing dataset...")
    try:
        dataset = SOCOFingSiameseDataset(data_dir=args.data_dir, transform=transform, num_pairs=args.num_pairs)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the SOCOFing dataset is located in the data/raw directory.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SiameseNetwork(embedding_dim=128, pretrained=True).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs with {args.num_pairs} pairs...")
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch} Average Loss: {avg_loss:.4f}")

    print(f"Training complete. Saving model to {args.save_path}...")
    torch.save(model.state_dict(), args.save_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
