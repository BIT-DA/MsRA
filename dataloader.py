import torch
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms


class Getset(data.Dataset):
    def __init__(self, dataset='', mode='', s_t='', source='', target='', c_cls='', train_test='',transform=None):
        super(Getset, self).__init__()
        self.resize = 256
        self.dataset = dataset
        self.center_crop = 224
        self.mode = mode
        self.s_t = s_t
        self.source = source
        self.target = target
        self.c_cls = c_cls
        self.train_test = train_test
        self.transform = transform
        self.txt_path = ''
        if self.dataset == "OfficeHomeDataset":
            if self.mode == 'MsRA_training':
                if self.train_test == 'train':
                    self.txt_path = os.path.join('.', 'dataset', self.dataset,
                                                 self.s_t + '_' + self.source + '_' + self.target + '_' + self.c_cls + '_' + self.train_test + '.txt')
                    self.transform = transforms.Compose([
                        transforms.Resize(self.resize),
                        transforms.CenterCrop((self.center_crop, self.center_crop)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=(30)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.train_test == 'test':
                    self.txt_path = os.path.join('.', 'dataset', self.dataset,
                                                 'test_' + self.source + '_' + self.target + '_' + self.c_cls + '_' + self.train_test + '.txt')
                    self.transform = transforms.Compose([
                        transforms.Resize(self.resize),
                        transforms.CenterCrop((self.center_crop, self.center_crop)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

        f = open(self.txt_path, 'r')
        data_list = f.readlines()
        f.close()
        self.n_data = len(data_list)
        self.img_names = []
        self.img_labels = []
        for _data in data_list:
            self.img_names.append(_data[:-3])
            self.img_labels.append(_data[-2])
        self.img_path_root = os.path.join('.', 'dataset', self.dataset)

    def __getitem__(self, item):
        img_names, labels = self.img_names[item], self.img_labels[item]
        img_ = Image.open(os.path.join(self.img_path_root, img_names)).convert('RGB')
        img_ = self.transform(img_)
        labels = int(labels)
        return img_, labels

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    dataset_train_s = Getset(dataset='OfficeHomeDataset', mode='MsRA_training', s_t='s', source='Product', target='Clipart',
                             c_cls='Bike',train_test='train')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_train_s,
        batch_size=5,
        shuffle=True,
        num_workers=2
    )
    data_iter = iter(dataloader)
    data = data_iter.__next__()
    img, label = data
    print(label)
