from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import os
import torch

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_longtailed_dataset(Dataset): 
    def __init__(self, dataset, imb_type, imb_factor, root_dir, transform, mode): 
        self.transform = transform
        self.mode = mode
        
        if dataset=='cifar10':
            base_folder = 'cifar-10-batches-py'
        elif dataset=='cifar100':
            base_folder = 'cifar-100-python'
        file_path = os.path.join(root_dir, base_folder)
        
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%file_path)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%file_path)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(file_path,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%file_path)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.data = train_data
            self.targets = np.array(train_label)
            self.cls_num = 10 if dataset == 'cifar10' else 100

            # Generate long-tailed imbalanced data
            self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            os.makedirs(os.path.join(file_path, 'longtail_file'), exist_ok=True)
            imb_file = os.path.join(file_path, 'longtail_file', 'cifar' + str(self.cls_num) + '_' + imb_type + '_' + str(imb_factor))
            self.gen_imbalanced_data(self.img_num_list, imb_file)
            
            # No noise injection here, unlike the original implementation
            self.train_data = self.data
            self.train_label = self.targets

    def __getitem__(self, index):
        if self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        else:  # train mode
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
           
    def __len__(self):
        if self.mode=='test':
            return len(self.test_data)
        else:
            return len(self.train_data)        

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:  # No imbalance (uniform distribution)
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, imb_file):
        # Initialize num_per_cls_dict regardless of whether we're loading saved indices or not
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            
        if os.path.exists(imb_file):
            imb_sample = json.load(open(imb_file,"r"))
        else:
            imb_sample = []
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                imb_sample.extend(selec_idx)
            imb_sample = np.array(imb_sample).tolist()
            print("save imbalanced data indices to %s ..." % imb_file)     
            json.dump(imb_sample, open(imb_file, 'w'))
        imb_sample = np.array(imb_sample)
        self.data = self.data[imb_sample]
        self.targets = self.targets[imb_sample]
    
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class cifar_longtailed_dataloader():  
    def __init__(self, dataset, imb_type, imb_factor, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
                
    def run(self, mode):
        if mode=='train':
            train_dataset = cifar_longtailed_dataset(
                dataset=self.dataset, 
                imb_type=self.imb_type, 
                imb_factor=self.imb_factor, 
                root_dir=self.root_dir, 
                transform=self.transform_train, 
                mode="train"
            )
            trainloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )            
            return trainloader, train_dataset.get_cls_num_list()
        
        elif mode=='test':
            test_dataset = cifar_longtailed_dataset(
                dataset=self.dataset, 
                imb_type=self.imb_type, 
                imb_factor=self.imb_factor, 
                root_dir=self.root_dir, 
                transform=self.transform_test, 
                mode="test"
            )
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )          
            return test_loader