import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class KeyPoint3DCNN(nn.Module):
    def __init__(self):
        super(KeyPoint3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output keypoint coordinates
        return x

def extract_roi(volume, center, size):
    z_center, y_center, x_center = center

    # Calculate the start and end indices for the ROI
    # The ROI is defined by the center and the size
    z_start = max(z_center - size[0] // 2, 0)
    y_start = max(y_center - size[1] // 2, 0)
    x_start = max(x_center - size[2] // 2, 0)

    z_end = min(z_center + size[0] // 2, volume.shape[0])  # depth
    y_end = min(y_center + size[1] // 2, volume.shape[1])  # height
    x_end = min(x_center + size[2] // 2, volume.shape[2])  # width

    if z_start >= z_end or y_start >= y_end or x_start >= x_end:
        raise ValueError(f"Invalid ROI: The specified ROI is out of bounds. "
                         f"z_start={z_start}, z_end={z_end}, "
                         f"y_start={y_start}, y_end={y_end}, "
                         f"x_start={x_start}, x_end={x_end}")

    return volume[z_start:z_end, y_start:y_end, x_start:x_end]


if __name__ == '__main__':
    # Generate random inputs and targets
    num_samples = 100
    input_size = (1, 32, 32, 32)  # 1 channel, 32x32x32 volume
    keypoint_coords = np.random.rand(num_samples, 3) * 32  # Generate 3D coordinates with values between 0 and 32

    inputs = torch.rand(num_samples, *input_size)  # Gerate random inputs
    targets = torch.tensor(keypoint_coords, dtype=torch.float32)

    # Define the ROI
    center = (16, 16, 16) 
    size = (8, 8, 8) 

    roi_inputs = []
    for i in range(num_samples):
        try:
            roi = extract_roi(inputs[i][0].numpy(), center, size)
            roi_inputs.append(torch.tensor(roi).unsqueeze(0))
        except ValueError as e:
            print(f"Error extracting ROI for sample {i}: {e}")
            roi_inputs.append(torch.zeros(1, size[0], size[1], size[2])) 

    roi_inputs = torch.stack(roi_inputs)
    print(roi_inputs.shape) 

    dataset = TensorDataset(roi_inputs, targets)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KeyPoint3DCNN()
    # output = model(roi_inputs)
    # print(output.shape)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



