import os
import os.path as osp
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')   # “Anti-Ghost” backend: no windows
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from torchvision import transforms


class CSSrcDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace("leftImg8bit", "gtFine_labelIds")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        # label = torch.LongTensor(np.array(label).astype('int32'))
        
        # 2) pull out raw IDs
        label_np = np.array(label).astype(np.int32)

        # 3) map those 0–33 IDs → 0–18 (or 255) trainIDs
        ID_TO_TRAINID = {
            0:255,1:255,2:255,3:255,4:255,5:255,6:255,
            7:0,8:1,9:255,10:255,11:2,12:3,13:4,
            14:255,15:255,16:255,17:5,18:255,19:6,
            20:7,21:8,22:9,23:10,24:11,25:12,
            26:13,27:14,28:15,29:255,30:255,31:16,
            32:17,33:18
        }
        train_label = 255 * np.ones_like(label_np, dtype=np.uint8)
        for orig_id, train_id in ID_TO_TRAINID.items():
            train_label[label_np == orig_id] = train_id

        # 4) back to tensor
        label = torch.from_numpy(train_label).long()
        
        
        
        return image, label, np.array(size), name


if __name__ == '__main__':
    dst = CSSrcDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
