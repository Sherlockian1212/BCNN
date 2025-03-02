from torchvision import transforms
from PIL import Image
import torch
from pycocotools.coco import COCO
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transform=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Lấy category_id đầu tiên (giả sử mỗi ảnh có 1 đối tượng duy nhất)
        if len(anns) > 0:
            labels = torch.tensor(anns[0]['category_id']).long()
        else:
            labels = torch.tensor(0).long()  # Class 0 nếu không có đối tượng

        if self.transform:
            img = self.transform(img)

        return img, labels
