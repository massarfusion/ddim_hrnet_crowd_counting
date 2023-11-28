from datasets.shtech import Shanghaitech
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch


if __name__ == "__main__":
    
    t= torch.tensor([ 65, 751,  15,], device='cpu')
    
    
    dataset = Shanghaitech(
        root_path=r"C:\DM-Count\DATA\shanghairaw\ShanghaiTech\part_A",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    train_loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    for i, (x, y) in enumerate(train_loader):
        # Convert tensors to numpy arrays
        x = x.numpy()
        y = y.numpy()
        
        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for i in range(1):
            axs[i].imshow(x[i].transpose(1, 2, 0))  # Plot RGB image
            
            # Convert density map to heatmap
            heatmap = axs[i].imshow(y[i, 0], cmap='hot', alpha=0.85)
            fig.colorbar(heatmap)
        
        plt.show()
        
        
    
    