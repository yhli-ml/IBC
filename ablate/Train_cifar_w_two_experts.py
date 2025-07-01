from __future__ import print_function
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from utils.PreResNet import *
from utils.KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import utils.dataloader_cifar as dataloader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import faiss
from sklearn.metrics import silhouette_score
from PIL import Image
import cv2
import torchvision.transforms as transforms
import pandas as pd
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int, required=True)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/nas/datasets', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.1, type=float)
parser.add_argument('--noise_mode',  default='imb')
parser.add_argument('--noise_ratio', default=0.2, type=float, help='noise ratio')
parser.add_argument('--arch', default='resnet18', type=str, help='resnet18')
parser.add_argument('-w', '--warm_up', default=30, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)
parser.add_argument('--beta', default=0.99, type=float, help='smoothing factor')
parser.add_argument('--phi', default=1.005, type=float, help='parameter for dynamic threshold')
parser.add_argument('--sample_rate', default=5, type=int, help='sampling rate of SFA')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
parser.add_argument('--k', default=0.5, type=float)
parser.add_argument('--epsilon', default=0.2, type=float)

file_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
args.pretrained=f"../moco_ckpt/PreActResNet18/{args.dataset}_exp_{args.imb_factor}/checkpoint_2000.pth.tar"

print(args)

def set_seed():
    torch.cuda.set_device(args.gpuid)
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed()

# Warm Up
def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        feats, logits1 = net(inputs, return_features=True)
        loss = CEloss(logits1, labels)

        logits2 = net.classify2(feats)
        loss_BCE = balanced_softmax_loss(labels, logits2, torch.tensor(dataloader.dataset.real_img_num_list), "mean")

        logits3 = net.classify3(feats)
        loss_BCE_3 = balanced_softmax_loss(labels, logits3, torch.tensor(dataloader.dataset.real_img_num_list), "mean", tau=0.5)

        L = loss + loss_BCE
        L.backward()
        optimizer.step()

        if batch_idx%50==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, current_labels, tmp_img_num_list, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx):
    net.train()
    net2.eval() # fix one network and train the other
    img_num_list_via_soft_label=torch.zeros(args.num_class)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2, index_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = current_labels[index_x]
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            targets_x_tl = torch.zeros_like(px, dtype=float)
            targets_x_hd = torch.zeros_like(px, dtype=float)
            targets_x_md = torch.zeros_like(px, dtype=float)

            # Long-tailed label refinement

            # target for tail expert
            # 1. Reduce confidence for the noisy label
            targets_x_tl[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_hd[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_md[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon

            # 2. Increase confidence of the k nearest class prototypes in feature space
            for i in range(batch_size): 
                neighbors = tail_nearest_neighbors_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_tl[i, neighbors] += args.epsilon / num_neighbors

            targets_x_tl = targets_x_tl / targets_x_tl.sum(dim=1, keepdim=True)
            targets_x_tl = targets_x_tl.detach()

            # target for head expert
            # for i in range(batch_size):
            #     neighbors = head_nearest_neighbors_idx[index_x[i]]
            #     num_neighbors = len(neighbors)
            #     if num_neighbors > 0:
            #         targets_x_hd[i, neighbors] += args.epsilon / num_neighbors
            
            # targets_x_hd = targets_x_hd / targets_x_hd.sum(dim=1, keepdim=True)
            targets_x_hd = labels_x.detach()

            # target for medium expert
            for i in range(batch_size):
                neighbors = medium_nearest_neighbors_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_md[i, neighbors] += args.epsilon / num_neighbors
            
            targets_x_md = targets_x_md / targets_x_md.sum(dim=1, keepdim=True)
            targets_x_md = targets_x_md.detach()

            for i in range(args.num_class):
                img_num_list_via_soft_label[i] += targets_x_tl.sum(dim=0)[i].cpu().numpy()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets_tl = torch.cat([targets_x_tl, targets_x_tl, targets_u, targets_u], dim=0)
        all_targets_hd = torch.cat([targets_x_hd, targets_x_hd, targets_u, targets_u], dim=0)
        all_targets_md = torch.cat([targets_x_md, targets_x_md, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a_tl, target_b_tl = all_targets_tl, all_targets_tl[idx]
        target_a_hd, target_b_hd = all_targets_hd, all_targets_hd[idx]
        target_a_md, target_b_md = all_targets_md, all_targets_md[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target_hd = l * target_a_hd + (1 - l) * target_b_hd
        mixed_target_tl = l * target_a_tl + (1 - l) * target_b_tl
        mixed_target_md = l * target_a_md + (1 - l) * target_b_md
                
        feats, logits = net(mixed_input, return_features=True)
        logits2 = net.classify2(feats)
        logits3 = net.classify3(feats)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        logits2_x = logits2[:batch_size*2]
        logits2_u = logits2[batch_size*2:]
        logits3_x = logits3[:batch_size*2]
        logits3_u = logits3[batch_size*2:]
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target_hd[:batch_size*2], logits_u, mixed_target_hd[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        loss_BCE_x, loss_BCE_u = balanced_softmax_loss_Semi(logits2_x, mixed_target_tl[:batch_size*2], logits2_u, mixed_target_tl[batch_size*2:], tmp_img_num_list)
        loss_BCE_x_md, loss_BCE_u_md = balanced_softmax_loss_Semi(logits3_x, mixed_target_md[:batch_size*2], logits3_u, mixed_target_md[batch_size*2:], tmp_img_num_list, tau=0.5)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + loss_BCE_x + lamb * loss_BCE_u + loss_BCE_x_md + lamb * loss_BCE_u_md + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%50==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g_%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.imb_type, args.imb_factor, args.noise_mode, args.noise_ratio, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()
    return img_num_list_via_soft_label


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    total_preds = []
    total_targets = []
    
    many_shot_preds, many_shot_targets = [], []
    medium_shot_preds, medium_shot_targets = [], []
    few_shot_preds, few_shot_targets = [], []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)           
            outputs1 = net1.classify2(feats1)
            outputs11 = net1.classify3(feats1)
            outputs2 = net2.classify2(feats2)
            outputs21 = net2.classify3(feats2)
            outputs = outputs1 + outputs2 + outputs01 + outputs02 + outputs11 + outputs21

            _, predicted = torch.max(outputs, 1)            

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            total_preds.append(predicted)
            total_targets.append(targets)
            
            for i in range(len(targets)):
                if targets[i].item() in many_shot_classes:
                    many_shot_preds.append(predicted[i])
                    many_shot_targets.append(targets[i])
                elif targets[i].item() in medium_shot_classes:
                    medium_shot_preds.append(predicted[i])
                    medium_shot_targets.append(targets[i])
                elif targets[i].item() in few_shot_classes:
                    few_shot_preds.append(predicted[i])
                    few_shot_targets.append(targets[i])

    acc = 100. * correct / total

    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    
    cls_acc = [round(100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / (total_targets == i).sum().item(), 2) 
               for i in range(args.num_class)]
    
    def calculate_category_accuracy(preds, targets):
        if len(targets) == 0:
            return 0
        correct_preds = (torch.tensor(preds) == torch.tensor(targets)).sum().item()
        return 100. * correct_preds / len(targets)

    many_shot_acc = calculate_category_accuracy(many_shot_preds, many_shot_targets)
    medium_shot_acc = calculate_category_accuracy(medium_shot_preds, medium_shot_targets)
    few_shot_acc = calculate_category_accuracy(few_shot_preds, few_shot_targets)

    result_dict = {
        "overall_accuracy": acc,
        "class_accuracy": cls_acc,
        "many_shot_accuracy": many_shot_acc,
        "medium_shot_accuracy": medium_shot_acc,
        "few_shot_accuracy": few_shot_acc
    }

    return result_dict


def eval_train(model, cfeats_EMA, cfeats_sq_EMA):
    model.eval()
    total_features = torch.zeros((num_all_img, feat_size))  # save sample features
    total_labels = torch.zeros(num_all_img).long()  # save sample labels
    tmp_img_num_list = torch.zeros(args.num_class)  # compute N_k from clean sample set
    pred = np.zeros(num_all_img, dtype=bool)  # clean probability
    confs_BS = torch.zeros(num_all_img)  # confidence from ABC
    confs_BS_max = torch.zeros(num_all_img)  # max_confidence from ABC
    mask = torch.zeros(num_all_img).bool()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, outputs = model(inputs, return_features=True)
            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
            confs_BS_max[index] = probs2.max(dim=1)[0].cpu()
            
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]

    total_features = total_features.cuda()
    total_labels = total_labels.cuda()

    # Instant Centroid Estimation
    tau = 1 / args.num_class
    sampled_class = []  # classes that need to do SFA 
    for i in range(args.num_class):
        idx_selected = (confs_BS[idx_class[i]] > tau * args.phi ** epoch).nonzero(as_tuple=True)[0]
        idx_selected = idx_class[i][idx_selected]
        mask[idx_selected] = True
        if (idx_selected.size(0) > 300):
            sampled_class.append(i)
        
    remained_class = list(set(range(args.num_class)) - set(sampled_class))

    # stochastic feature averaging
    if epoch == warm_up + 1:
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=None)  # (10, 512)
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)  # (10, 512)
        refined_cfeats['cl2ncs'] = np.nan_to_num(refined_cfeats['cl2ncs'], nan=0.0) # in case of none selected (especially tail class)
        cfeats_EMA = args.beta * cfeats_EMA + (1 - args.beta) * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = args.beta * cfeats_sq_EMA + (1 - args.beta) * refined_cfeats['cl2ncs'] ** 2
    
    # -----------------------------------------
    # k nearest neighbor prototype searching...
    def find_knn_with_faiss_cosine(features, prototypes, k):
        # Step 1: Normalize the features and prototypes to unit length
        features = features - features.mean(dim=1, keepdim=True)  # Center the features by subtracting the mean
        
        # L2 normalization
        norm_features = torch.norm(features, p=2, dim=1, keepdim=True)
        normalized_features = features / norm_features

        # Convert features and prototypes to numpy arrays for FAISS
        normalized_features_np = normalized_features.cpu().numpy().astype(np.float32)

        # Step 2: Create FAISS index for prototypes
        index = faiss.IndexFlatIP(prototypes.shape[1])  # Use inner product (dot product) for cosine similarity
        index.add(prototypes)  # Add prototypes to the index

        # Step 3: Search for the k-nearest neighbors
        distances, indices = index.search(normalized_features_np, k)  # distances are the inner product, indices are the k nearest neighbors

        return normalized_features_np, indices  # The indices of the k nearest prototypes for each sample in features

    k = int(args.num_class * args.k)  # Number of nearest neighbors
    normalized_features_np, nearest_neighbors_idx = find_knn_with_faiss_cosine(total_features, cfeats_EMA, k)
    # -----------------------------------------

    # Generate the new nearest neighbors index by filtering out head class neighbors
    head_nearest_neighbors_idx = []
    medium_nearest_neighbors_idx = []
    tail_nearest_neighbors_idx = []

    for i in range(nearest_neighbors_idx.shape[0]):
        # Get the indices of k nearest neighbors for the current sample
        current_neighbors = nearest_neighbors_idx[i]
        # Keep only the neighbors that belong to tail classes
        head_neighbors = [label for label in current_neighbors if label in many_shot_classes]
        medium_neighbors = [label for label in current_neighbors if label in medium_shot_classes]
        tail_neighbors = [label for label in current_neighbors if label in few_shot_classes]
        # If the number of valid neighbors is less than k, we keep only the top k (in this case, it shouldn't happen)
        head_nearest_neighbors_idx.append(head_neighbors[:k])
        medium_nearest_neighbors_idx.append(medium_neighbors[:k])
        tail_nearest_neighbors_idx.append(tail_neighbors[:k])

    # Convert the new nearest neighbors index to a numpy array
    head_nearest_neighbors_idx = np.array(head_nearest_neighbors_idx, dtype=object)
    medium_nearest_neighbors_idx = np.array(medium_nearest_neighbors_idx, dtype=object)
    tail_nearest_neighbors_idx = np.array(tail_nearest_neighbors_idx, dtype=object)
    # nearest_neighbors_idx will be a matrix of shape (num_samples, k)
    # where each row contains the indices of the k most similar prototypes for a sample in total_features
    df = pd.DataFrame(head_nearest_neighbors_idx)
    save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/knn_head_{epoch}.csv'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name, index=False)
    df = pd.DataFrame(medium_nearest_neighbors_idx)
    save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/knn_medium_{epoch}.csv'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name, index=False)
    df = pd.DataFrame(tail_nearest_neighbors_idx)
    save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/knn_tail_{epoch}.csv'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name, index=False)
    # -----------------------------------------
    # compute silhouette score
    silhouette_avg = silhouette_score(normalized_features_np, noisy_labels)
    save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/silhouette_avg_{epoch}.np'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    df.to_csv(save_name, index=False)
    # -----------------------------------------
    # sample centers from gaussion postier
    refined_ncm_logits = torch.zeros((num_all_img, args.num_class)).cuda()
    for i in range(args.sample_rate):
        mean = cfeats_EMA
        std = np.sqrt(np.clip(cfeats_sq_EMA - mean ** 2, 1e-30, 1e30))
        eps = np.random.normal(size=mean.shape)
        cfeats = mean + std * eps

        refined_cfeats['cl2ncs'][sampled_class] = cfeats[sampled_class]
        refined_cfeats['cl2ncs'][remained_class] = mean[remained_class]

        ncm_classifier.update(refined_cfeats, device=args.gpuid)
        refined_ncm_logits += ncm_classifier(total_features, None)[0]

    # 'prob' describes the probability of each sample belonging to clean/noisy category
    prob = get_gmm_prob(refined_ncm_logits, total_labels)
    # -----------------------------------------
    for i in range(args.num_class):
        pred[idx_class[i]] = (prob[idx_class[i]] > args.p_threshold)
        tmp_img_num_list[i] = np.sum(pred[idx_class[i]])

    print('Number of clean samples selected per class: ', tmp_img_num_list)
    correct_predictions = ((torch.from_numpy(pred) == (noisy_labels == clean_labels)) & torch.from_numpy(pred)).sum()
    total_samples = len(noisy_labels)
    precision = correct_predictions / torch.from_numpy(pred).sum()
    cls_prec = [
        round(100. * ((torch.from_numpy(pred) == (clean_labels == noisy_labels)) & (clean_labels == i) & torch.from_numpy(pred)).sum().item() / tmp_img_num_list[i].item(), 2) 
        if tmp_img_num_list[i].item() != 0 else 'nan'
        for i in range(args.num_class)
    ]
    actual_clean = (noisy_labels == clean_labels).sum()

    recall = correct_predictions / actual_clean
    cls_recall = [
    round(100. * ((torch.from_numpy(pred) == (clean_labels == noisy_labels)) & (clean_labels == i) & torch.from_numpy(pred)).sum().item() 
        / ((clean_labels == i) & (noisy_labels == clean_labels)).sum().item(), 2) 
        if ((clean_labels == i) & (noisy_labels == clean_labels)).sum().item() != 0 else 'nan'
        for i in range(args.num_class)
    ]
    print("Precision: %.4f%% %s\nRecall: %.4f%% %s" %(precision.item(), str(cls_prec), recall.item(), str(cls_recall)))
    # ----------------------------------------------------------------
    # Feature Visualization: Select classes for head, medium, and tail        
    if epoch % 10 == 0:
        # Use t-SNE to reduce dimensions to 2D for visualization
        tsne = TSNE(n_components=2, random_state=0)
        features_2d = tsne.fit_transform(normalized_features_np)

        save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/featur_2d_{epoch}.npy'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, features_2d)

    return pred, confs_BS_max, tmp_img_num_list, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx, silhouette_avg

def get_gmm_prob(ncm_logits, total_labels):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # normalization, note that the logits are all negative
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()

def get_knncentroids(feats=None, labels=None, mask=None):

    feats = feats.cpu().numpy()
    labels = labels.cpu().numpy()

    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype('bool')
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()
            
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_==i) & mask_], axis=0))
        return np.stack(centroids)

    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels, mask)
    
    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,
            'cl2ncs': cl2n_centers}

# Function to generate and visualize Class Activation Map (CAM)
def generate_and_visualize_cam(model, img_path, target_layer, device='cuda'):
    """
    Generate and visualize Class Activation Map (CAM) for a given image and model.

    Args:
        model (torch.nn.Module): The trained model (e.g., ResNet).
        img_path (str): Path to the input image.
        target_layer (torch.nn.Module): The layer from which to extract features (e.g., the last conv layer).
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        None
    """
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Set the model to evaluation mode
    model.eval()

    # Hook to get the feature map from the target layer
    def hook_fn(module, input, output):
        global feature_map
        feature_map = output  # Store the feature map

    # Attach the hook to the target layer
    target_layer.register_forward_hook(hook_fn)

    # Forward pass through the model
    output = model(input_tensor)

    # Get the predicted class
    _, pred_class = output.max(1)

    # Zero gradients and backward pass for the predicted class
    model.zero_grad()
    output[0, pred_class].backward()

    # Step 1: Compute the weights (global average pooling of the gradients)
    gradients = feature_map.grad
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling of gradients

    # Step 2: Compute the CAM
    cam = torch.sum(weights * feature_map, dim=1, keepdim=True)
    cam = F.relu(cam)  # Apply ReLU to keep positive values

    # Step 3: Upsample CAM to the size of the input image
    cam = cam.squeeze().cpu().detach().numpy()  # Remove batch dimension and convert to NumPy array
    cam = np.maximum(cam, 0)  # Remove negative values
    cam = cv2.resize(cam, (img.size[0], img.size[1]))  # Resize CAM to the input image size

    # Step 4: Visualize the CAM over the image
    img = np.array(img)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.imshow(cam, alpha=0.6, cmap='jet')  # Overlay the CAM on the image
    plt.title(f'Class Activation Map for Class {pred_class.item()}')
    plt.colorbar()

    # Optionally save the figure
    save_name = f'./CAM_demo/{file_name}_cam_class_{pred_class.item()}.png'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.savefig(save_name)
    plt.close()

    return cam  # Optionally return the CAM for further use

def feature_extractor(model, eval_loader):    
    model.eval()
    total_features = torch.zeros((num_all_img, feat_size))  # save sample features

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, outputs = model(inputs, return_features=True)
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
    return total_features.numpy()

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def create_model():
    model = PreActResNet18_3(num_classes=args.num_class)
    model = model.cuda()
    return model

def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1.0):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def balanced_softmax_loss_Semi(logits_x, targets_x, logits_u, targets_u, sample_per_class, tau=1.0):
    spc = sample_per_class.type_as(logits_x)

    spc_x = spc.unsqueeze(0).expand(logits_x.shape[0], -1)
    logits_x = logits_x + tau * spc_x.log()
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    spc_u = spc.unsqueeze(0).expand(logits_u.shape[0], -1)
    logits_u = logits_u + tau * spc_u.log()
    probs_u = torch.softmax(logits_u, dim=1)
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu

if args.warm_up is not None:
    warm_up = args.warm_up
else:
    warm_up = 30


if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')

loader = dataloader.cifar_dataloader(args.dataset, imb_type=args.imb_type, imb_factor=args.imb_factor, noise_mode=args.noise_mode, noise_ratio=args.noise_ratio,\
    batch_size=args.batch_size, num_workers=5, root_dir=args.data_path)
args.num_class = 100 if args.dataset == 'cifar100' else 10
feat_size = 512
ncm_classifier = KNNClassifier(feat_size, args.num_class)
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long()
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()
real_img_num_list = torch.Tensor(loader.run('warmup').dataset.real_img_num_list).long()

df = pd.DataFrame(noisy_labels)
save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/noisy_labels.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=False)

df = pd.DataFrame(clean_labels)
save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/clean_labels.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=False)

df = pd.DataFrame(real_img_num_list)
save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/real_img_num_list.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=False)

warmup_trainloader = loader.run('warmup')
eval_loader = loader.run('eval_train')   
num_all_img = len(warmup_trainloader.dataset)
idx_class = []  # index of sample in each class
for i in range(args.num_class):
    idx_class.append((torch.tensor(warmup_trainloader.dataset.noise_label) == i).nonzero(as_tuple=True)[0])
#-----------------------------------------------------  
class_counts = [0] * args.num_class
for _, targets, index in warmup_trainloader:
    for target in targets:
        class_counts[target.item()] += 1
# Get the class indices and their corresponding counts
sorted_class_counts = sorted(enumerate(class_counts), key=lambda x: x[1], reverse=True)

# Calculate the boundary points for the top 30%, middle 40%, and bottom 30% classes
total_classes = len(class_counts)
top_30_percent = int(total_classes * 0.3)
bottom_30_percent = int(total_classes * 0.3)

# Sort the classes by their counts in descending order
sorted_classes = [class_idx for class_idx, count in sorted_class_counts]

# Select the top 30% as many-shot classes, bottom 30% as few-shot classes, and the remaining 40% as medium-shot classes
many_shot_classes = set(sorted_classes[:top_30_percent])
few_shot_classes = set(sorted_classes[-bottom_30_percent:])
medium_shot_classes = set(sorted_classes[top_30_percent: -bottom_30_percent])
#--------------------------------------------------
cfeats_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_EMA2 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA2 = np.zeros((args.num_class, feat_size))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

# load moco pretrained weights
if os.path.isfile(args.pretrained):
    print("=> loading checkpoint '{}'".format(args.pretrained))
    checkpoint = torch.load(args.pretrained, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            # remove prefix
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    args.start_epoch = 0
    msg = net1.load_state_dict(state_dict, strict=False)
    msg2 = net2.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias", "linear3.weight", "linear3.bias"} #
    assert set(msg2.missing_keys) == {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias", "linear3.weight", "linear3.bias"} #

    print("=> loaded pre-trained model '{}'".format(args.pretrained))
else:
    print("=> no checkpoint found at '{}'".format(args.pretrained))

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


if args.resume is not None:
    resume_path = f'./checkpoint/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{args.resume}.pth'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    print(f'| Loading model from {resume_path}')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        prob1 = ckpt['prob1']
        prob2 = ckpt['prob2']
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        start_epoch = 1
else:
    start_epoch = 1

CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()

acc_list=[]
best_acc = 0
best_epoch = 0
best_model = None
for epoch in range(start_epoch, args.num_epochs + 1):   
    lr = args.lr
    if epoch > 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch <= warm_up:
        warmup_trainloader = loader.run('warmup')

        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader) 
   
    else:
        current_labels1 = noisy_labels
        current_labels2 = noisy_labels

        pred1, prob1, tmp_img_num_list1, head_nearest_neighbors_idx1, medium_nearest_neighbors_idx1, tail_nearest_neighbors_idx1, silhouette_avg1 = eval_train(net1, cfeats_EMA1, cfeats_sq_EMA1)  # sample selection from net1
        pred2, prob2, tmp_img_num_list2, head_nearest_neighbors_idx2, medium_nearest_neighbors_idx2, tail_nearest_neighbors_idx2, silhouette_avg2 = eval_train(net2, cfeats_EMA2, cfeats_sq_EMA2)  # sample selection from net2

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, current_labels2, tmp_img_num_list2, head_nearest_neighbors_idx1, medium_nearest_neighbors_idx1, tail_nearest_neighbors_idx1) # train net1  
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, current_labels1, tmp_img_num_list1, head_nearest_neighbors_idx2, medium_nearest_neighbors_idx2, tail_nearest_neighbors_idx2) # train net2      

    acc_dict = test(epoch, net1, net2)
    test_acc = acc_dict['overall_accuracy']
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_model = {
            'net1': net1.state_dict(),
            'net2': net2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'acc_dict': acc_dict
        }

    acc_list.append(acc_dict)
    
    if epoch > warm_up:
        df = pd.DataFrame(pred2)
        save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/prediction.csv'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        df.to_csv(save_name, index=False)

    print(f"\n| Test Epoch #{epoch}\t Accuracy: {test_acc:6.2f}%\t Best Acc: {best_acc:6.2f}\t at Epoch {best_epoch}.")
    print(acc_dict['class_accuracy'])
    print(f"Many Shot Accuracy: {acc_dict['many_shot_accuracy']:6.2f}%")
    print(f"Medium Shot Accuracy: {acc_dict['medium_shot_accuracy']:6.2f}%")
    print(f"Few Shot Accuracy: {acc_dict['few_shot_accuracy']:6.2f}%")

    if epoch in [warm_up, args.num_epochs]:
        save_name = f'./checkpoint/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{epoch}.pth'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        print(f'| Saving model to {save_name}')

        ckpt = {'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'prob1': prob1 if 'prob1' in dir() else None,
                'prob2': prob2 if 'prob2' in dir() else None}
        torch.save(ckpt, save_name)


df = pd.DataFrame(acc_list)
save_name = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}/Acc.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=True)

# Save the best model after training completes
if best_model is not None:
    best_model_path = f'./checkpoint/{file_name}/best_model.pth'
    print(f"Saving the best model at epoch {best_epoch} with accuracy {best_acc:.2f}%")
    torch.save(best_model, best_model_path)    
