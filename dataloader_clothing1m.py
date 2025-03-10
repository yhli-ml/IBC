
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class clothing_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode,num_class, pred=[], probability=[]): 
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.cls_num = num_class
        num_samples=1000000
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if self.mode=='test':
            self.test_imgs = []  
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)                     
        else:
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for i , l in enumerate(lines):
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append((i,img_path))
            self.num_raw_example = len(train_imgs)                              
            random.shuffle(train_imgs)
            class_num = torch.zeros(self.cls_num)
            self.train_imgs = []
            for id_raw, impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append((id_raw,impath))
                    class_num[label]+=1
            random.shuffle(self.train_imgs)

            self.train_label_value = []

            for id_raw, img_path in self.train_imgs:
                label = self.train_labels[img_path]
                self.train_label_value.append(label)

            self.real_img_num_list = [0] * self.cls_num
            for i in range(len(self.train_label_value)):
                self.real_img_num_list[self.train_label_value[i]] += 1
            
            if self.mode == 'all':
                self.train_data = self.train_imgs
            else:                 
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                            
                self.train_data = self.train_imgs[pred_idx]
                self.pred_idx = pred_idx
                print("%s data has a size of %d"%(self.mode,len(self.train_data)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            img = Image.open(img_path).convert('RGB')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob, self.pred_idx[index]
        elif self.mode=='unlabeled':
            id_raw, img_path = self.train_imgs[index]
            img = Image.open(img_path).convert('RGB')   
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, self.pred_idx[index]
        elif self.mode=='all':
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            img = Image.open(img_path).convert('RGB') 
            img = self.transform(img)
            return img, target, index
        elif self.mode=='test':
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            img = Image.open(img_path).convert('RGB') 
            img = self.transform(img)
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
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
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, imb_file):
        if os.path.exists(imb_file):
            imb_sample = json.load(open(imb_file,"r"))
        else:
            imb_sample = []
            targets_np = np.array(self.targets, dtype=np.int64)
            classes = np.unique(targets_np)
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                imb_sample.extend(selec_idx)
            imb_sample = np.array(imb_sample).tolist()
            print("save imb labels to %s ..." % imb_file)     
            json.dump(imb_sample, open(imb_file, 'w'))
        imb_sample = np.array(imb_sample)
        self.data = self.data[imb_sample]
        self.targets = self.targets[imb_sample]
    
    def get_noisy_data(self, cls_num, noise_file, noise_mode, noise_ratio):
        train_label = self.targets
        
        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file,"r"))
        else:    #inject noise
            noise_label = []
            num_train = len(self.targets)
            idx = list(range(num_train))
            random.shuffle(idx)
            cls_num_list = self.img_num_list
            
            if noise_mode == 'sym':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = (random.randint(1, cls_num - 1) + train_label[i]) % cls_num
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:
                        noise_label.append(train_label[i])

            elif noise_mode == 'imb':
                num_noise = int(noise_ratio * num_train)
                noise_idx = idx[:num_noise]

                p = np.array([cls_num_list for _ in range(cls_num)])
                for i in range(cls_num):
                    p[i][i] = 0
                p = p / p.sum(axis=1, keepdims=True)
                for i in range(num_train):
                    if i in noise_idx:
                        newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                        assert newlabel != train_label[i]
                        noise_label.append(newlabel)
                    else:    
                        noise_label.append(train_label[i])

            noise_label = np.array(noise_label, dtype=np.int8).tolist()
            #label_dict['noisy_labels'] = noise_label
            print("save noisy labels to %s ..." % noise_file)     
            json.dump(noise_label, open(noise_file,"w"))

        self.clean_targets = self.targets[:]
        self.targets = noise_label

        for c1, c0 in zip(self.targets, self.clean_targets):
            if c1 != c0:
                self.img_num_list[c1] += 1
                self.img_num_list[c0] -= 1
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

        
class clothing_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])
    def run(self,mode,pred=[],prob=[],refine_labels=None, imb_factor=1):
        if mode=='warmup':
            all_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all", num_class=self.num_class)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode=='train':
            labeled_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="labeled",num_class=self.num_class,pred=pred,probability=prob)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",num_class=self.num_class,pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test', num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode=='eval_train':
            eval_dataset = clothing_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all', num_class=self.num_class, imb_factor=self.imb_factor)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader
