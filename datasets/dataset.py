import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import matplotlib.pylab as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class MapDataset(Dataset):
    def __init__(self, dir_data:list, train = True):
        """
        Load all data as Numpy array
        :param dir_data: directory of source images
        :param train: to perform augmentation or not
        """
        self.img = np.expand_dims(np.load(dir_data[0]), -1)  # MR image(s)
        self.target = np.expand_dims(np.load(dir_data[1]), -1)  # MR image(s)

        self.train = train

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        input_image = self.img[index]
        target_image = self.target[index]

        if self.train:
            augmentations = both_transform(image=input_image, image0=target_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]
        else:
            input_image = transform_only_input(image=input_image)["image"]
            target_image = transform_only_input(image=target_image)["image"]

        return {'mri':input_image, 'ct':target_image}


both_transform = A.Compose(
    [
        A.Resize(width=450, height=270, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=.5),
        A.Rotate(limit=20, p=0.8, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_CUBIC),
        A.GridDistortion(border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
        ToTensorV2()
    ],
        additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [

        A.Resize(width=450, height=270, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(height=256, width=256),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
        ToTensorV2(),
    ]
)



if __name__ == "__main__":
    dir_t2_train = r'../data/test_src1.npy'
    dir_ct_train = r'../data/test_src2.npy'
    dataset = MapDataset([dir_t2_train, dir_ct_train], train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        print(x.shape)
        plt.imshow(np.squeeze(x), cmap='gray')
        plt.figure()
        plt.imshow(np.squeeze(y), cmap='gray')
        plt.show()
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()