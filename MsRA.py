import torch.nn as nn
import torch.utils.data
from models import MsRa_model
import numpy as np
import argparse
from dataloader import *
from sklearn.metrics import roc_auc_score
from torch.optim import SGD


class MsRA():
    def __init__(self, args_):
        self.args = args_
        if args.cuda == -1:
            self.device = "cpu"
        else:
            torch.cuda.set_device(self.args.cuda)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCEWithLogitsLoss()
        self.c = None
        self.c = torch.zeros(self.c, device=self.device) if self.c is not None else None
        self.nu = 0.1

    def test(self, my_net):
        alpha = 0
        my_net = my_net.eval()

        # load test data
        dataset_test = Getset(mode=self.args.mode, dataset=self.args.dataset, source=self.args.source,
                              target=self.args.target, c_cls=self.args.c_cls, train_test='test')
        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        len_dataloader = len(dataloader_test)
        data_target_iter = iter(dataloader_test)

        counter = 0
        while counter < len_dataloader:
            data_target = data_target_iter.__next__()
            t_img, t_label = data_target
            input_img = t_img.to(self.device)
            _ , _ , feature = my_net(input_data=input_img, alpha=alpha)
            if counter == 0:
                test_feat = feature.detach().cpu().numpy()
                test_label = t_label
            else:
                test_feat = np.concatenate((test_feat, feature.detach().cpu().numpy()), 0)
                test_label = np.concatenate((test_label, t_label), 0)
            counter += 1

        # load source data
        dataset_train_s = Getset(mode=self.args.mode, dataset=self.args.dataset, s_t='s', source=self.args.source,
                                 target=self.args.target, c_cls=self.args.c_cls, train_test='train')
        dataloader_train_s = torch.utils.data.DataLoader(
            dataset=dataset_train_s,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        len_dataloader = len(dataloader_train_s)
        data_iter = iter(dataloader_train_s)
        counter = 0
        while counter < len_dataloader:
            data = data_iter.__next__()
            input, label = data
            input = torch.as_tensor(np.array(input), dtype=torch.float32).to(self.device)
            label_s = np.array(np.array(label))
            _, _, feature = my_net(input_data=input, alpha=alpha)
            if counter == 0:
                s_feat = feature.detach().cpu().numpy()
                s_label = label_s
            else:
                s_feat = np.concatenate((s_feat, feature.detach().cpu().numpy()), 0)
                s_label = np.concatenate((s_label, label_s), 0)
            counter += 1

        # load target data
        dataset_train_t = Getset(mode=self.args.mode, dataset=self.args.dataset, s_t='t',  source=self.args.source,
                                 target=self.args.target, c_cls=self.args.c_cls, train_test='train')
        dataloader_train_t = torch.utils.data.DataLoader(
            dataset=dataset_train_t,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        len_dataloader = len(dataloader_train_t)
        data_iter = iter(dataloader_train_t)
        counter = 0
        while counter < len_dataloader:
            data = data_iter.__next__()
            input, label = data
            input = torch.as_tensor(np.array(input), dtype=torch.float32).to(self.device)
            label_t = np.array(np.array(label))
            _, _, feature = my_net(input_data=input, alpha=alpha)
            if counter == 0:
                t_feat = feature.detach().cpu().numpy()
                t_label = label_t
            else:
                t_feat = np.concatenate((t_feat, feature.detach().cpu().numpy()), 0)
                t_label = np.concatenate((t_label, label_t), 0)
            counter += 1

        train_feat = np.concatenate((s_feat, t_feat), 0)
        self.c = np.mean(train_feat, axis=0)
        score = np.linalg.norm(test_feat - self.c, axis=1)
        auroc = roc_auc_score(test_label, score)
        return auroc

    def init_center_c(self,train_loader, net, alpha, net_res_dim, eps=0.1):
        n_samples = 0
        c = torch.zeros(net_res_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                class_output, domain_output, feature = net(inputs, alpha)
                n_samples += feature.shape[0]
                c += torch.sum(feature, dim=0)
        c /= n_samples

        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def get_radius(self,dist: torch.Tensor, nu: float):
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

    def train(self):
        # load source data
        dataset_train_s = Getset(mode=self.args.mode, dataset=self.args.dataset, s_t='s', source=self.args.source,
                                 target=self.args.target, c_cls=self.args.c_cls, train_test='train')
        dataloader_train_s = torch.utils.data.DataLoader(
            dataset=dataset_train_s,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        # load target data
        dataset_train_t = Getset(mode=self.args.mode, dataset=self.args.dataset, s_t='t', source=self.args.source,
                                 target=self.args.target, c_cls=self.args.c_cls, train_test='train')
        dataloader_train_t = torch.utils.data.DataLoader(
            dataset=dataset_train_t,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        len_dataloader = min(len(dataloader_train_s), len(dataloader_train_t))

        # load model
        my_net = MsRa_model().to(self.device)

        optimizer = SGD(my_net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        loss_domain = torch.nn.NLLLoss()

        # init c
        p = float(0 + self.args.epoch * len_dataloader) / self.args.epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        if self.c == None:
            self.c = self.init_center_c(dataloader_train_t, my_net, alpha, 256)

        auroc_list = []

        # training
        for epoch in range(self.args.epoch):
            data_source_iter = iter(dataloader_train_s)
            data_target_iter = iter(dataloader_train_t)

            i = 0
            while i < len_dataloader:

                p = float(i + epoch * len_dataloader) / self.args.epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training with source data
                data_source = data_source_iter.__next__()
                s_img, s_label = data_source

                my_net.zero_grad()

                input_img = torch.FloatTensor(self.args.batch_size, 3, self.args.image_size, self.args.image_size)
                domain_label = torch.zeros(self.args.batch_size)
                class_label = torch.zeros(self.args.batch_size)
                domain_label = domain_label.long()

                s_img = s_img.to(self.device)
                s_label = s_label.to(self.device)
                input_img = input_img.to(self.device)
                class_label = class_label.to(self.device)
                domain_label = domain_label.to(self.device)

                input_img.resize_as_(s_img).copy_(s_img)
                class_label.resize_as_(s_label).copy_(s_label)
                class_label = class_label.unsqueeze(1)

                class_output, domain_output,s_feature = my_net(input_data=input_img, alpha=alpha)
                center_s = torch.mean(s_feature,dim=0)
                svdd_loss_s = torch.sum(torch.norm((s_feature - center_s), dim=1, p=2)) / float(s_feature.shape[0])
                err_s_label = self.criterion(class_output, class_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training with target data
                data_target = data_target_iter.__next__()
                t_img, t_label = data_target

                input_img = torch.FloatTensor(self.args.image_size, 3, self.args.image_size, self.args.image_size)

                domain_label = torch.ones(self.args.batch_size)
                domain_label = domain_label.long()
                class_label = torch.zeros(self.args.batch_size)

                t_img = t_img.to(self.device)
                input_img = input_img.to(self.device)
                domain_label = domain_label.to(self.device)
                class_label = class_label.to(self.device)

                input_img.resize_as_(t_img).copy_(t_img)
                class_label.resize_as_(t_label).copy_(t_label)
                class_label = class_label.unsqueeze(1)

                class_output, domain_output ,t_feature = my_net(input_data=input_img, alpha=alpha)
                center_t = torch.mean(t_feature,dim=0)
                center_loss = torch.norm((center_s - center_t), p=2)
                svdd_loss_t = torch.sum(torch.norm((t_feature - center_t), dim=1, p=2)) / float(t_feature.shape[0])
                err_t_domain = loss_domain(domain_output, domain_label)
                err_t_label = self.criterion(class_output, class_label)
                err = (err_t_domain + err_s_domain + err_s_label  + err_t_label)+ 0.1 * (svdd_loss_s + svdd_loss_t + center_loss)
                err.backward()
                optimizer.step()

                i += 1

                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_label:%f, err_t_domain: %f, err_center： '
                      '%f, err_svdd: %f, total: %f' % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                         err_s_domain.cpu().data.numpy(), err_t_label.cpu().data.numpy(), err_t_domain.cpu().data.numpy(),
                         center_loss.cpu().data.numpy(),(svdd_loss_s + svdd_loss_t).cpu().data.numpy(), err.cpu().data.numpy()))

            auroc = self.test(my_net)
            auroc_list.append(auroc)

        return max(auroc_list)


if __name__ == "__main__":
    def get_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", type=str, default='MsRA_training')
        parser.add_argument("--dataset", type=str, default='OfficeHomeDataset')
        parser.add_argument("--source", type=str, default='Clipart')
        parser.add_argument("--target", type=str, default='Product')
        parser.add_argument("--c_cls", type=str, default='Bike')
        parser.add_argument("--feat_d", type=int, default=256)
        parser.add_argument("--epoch", type=int, default=35)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=0.5e-6)
        parser.add_argument("--save_models_path", type=str, default="./preserve/")
        parser.add_argument("--cuda", type=int, default=0)
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--resize", type=int, default=224)
        parser.add_argument("--center_crop", type=int, default=224)
        return parser.parse_args()
    args = get_arguments()

    best_auroc_list = []
    for i in range(10):
        model = MsRA(args)
        auroc = model.train()
        best_auroc_list.append(auroc)
    mean_auroc = np.mean(best_auroc_list) * 100
    std_auroc = np.std(best_auroc_list) * 100
    print('source: '+args.source)
    print('target: '+args.target)
    print('mean: '+str(mean_auroc) + ' ' + 'std: '+ str(std_auroc))
