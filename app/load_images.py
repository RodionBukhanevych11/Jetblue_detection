import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from io import BytesIO

def test_transform():
    return A.Compose([A.Resize(500, 500), ToTensor()])


class TestDataset(Dataset):
    def __init__(self, response, transforms=None):
        super().__init__()
        self.response = response
        self.transforms = transforms

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        image = np.asarray(Image.open(BytesIO(self.response.content)).convert("RGB")).astype(np.float32)
        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']
        image /= 255.0
        return image

def createDataset(response,batch_size=1):
    test_dataset = TestDataset(response, test_transform())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return test_loader