from __future__ import print_function
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from utils.PreResNet import *
from utils.KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import utils.dataloader_cifar as dataloader
from sklearn.manifold import TSNE
import faiss
from sklearn.metrics import silhouette_score
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# Training hyperparameters
parser.add_argument('-w', '--warm_up', default=30, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--gpuid', default=0, type=int, required=True)
parser.add_argument('--seed', default=123)

# Model hyperparameters
parser.add_argument('--arch', default='resnet18', type=str, help='model architechture')
parser.add_argument('-r', '--resume', default=None, type=int)

# Dataset hyperparameters
parser.add_argument('--data_path', default='/nas/datasets', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.1, type=float)
parser.add_argument('--noise_mode', default='imb')
parser.add_argument('--noise_ratio', default=0.2, type=float, help='noise ratio')
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')

# Method hyperparameters
parser.add_argument('--k', default=0.5, type=float)
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--alpha', default=4, type=float, help='parameter for mixmatch')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--beta', default=0.99, type=float, help='smoothing factor')
parser.add_argument('--phi', default=1.005, type=float, help='parameter for dynamic threshold')
parser.add_argument('--sample_rate', default=5, type=int, help='sampling rate of SFA')

file_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
args.pretrained = f"../moco_ckpt/PreActResNet18/{args.dataset}_exp_{args.imb_factor}/checkpoint_2000.pth.tar"

print(args)

def set_seed():
    """Set all random seeds for reproducibility"""
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

def warmup(epoch, net, optimizer, dataloader):
    """Warm up training for a single network"""
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    
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

        if batch_idx % 50 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(f'{args.dataset}:{args.imb_type}_{args.imb_factor}_{args.noise_mode}_{args.noise_ratio} | '
                            f'Epoch [{epoch:3d}/{args.num_epochs:3d}] Iter[{batch_idx+1:3d}/{num_iter:3d}]\t CE-loss: {loss.item():.4f}')
            sys.stdout.flush()

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, current_labels, 
          tmp_img_num_list, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx):
    """Train with co-teaching strategy between two networks"""
    net.train()
    net2.eval()  # Fix one network and train the other
    
    img_num_list_via_soft_label = torch.zeros(args.num_class)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        # Get unlabeled batch
        try:
            inputs_u, inputs_u2, index_u = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u = next(unlabeled_train_iter)
        
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = current_labels[index_x]
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        # Move tensors to GPU
        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + 
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/args.T)  # Temperature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # Normalize
            targets_u = targets_u.detach()

            # Label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            # Initialize targets for different experts
            targets_x_tl = torch.zeros_like(px, dtype=float)  # Tail expert
            targets_x_hd = torch.zeros_like(px, dtype=float)  # Head expert
            targets_x_md = torch.zeros_like(px, dtype=float)  # Medium expert

            # Long-tailed label refinement
            # 1. Reduce confidence for the noisy label
            targets_x_tl[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_hd[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_md[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon

            # 2. Increase confidence of the k nearest class prototypes in feature space
            # For tail expert
            for i in range(batch_size):
                neighbors = tail_nearest_neighbors_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_tl[i, neighbors] += args.epsilon / num_neighbors

            targets_x_tl = targets_x_tl / targets_x_tl.sum(dim=1, keepdim=True)
            targets_x_tl = targets_x_tl.detach()

            # For head expert - use original labels
            # Commented code is left unchanged as per original implementation
            # for i in range(batch_size):
            #     neighbors = head_nearest_neighbors_idx[index_x[i]]
            #     num_neighbors = len(neighbors)
            #     if num_neighbors > 0:
            #         targets_x_hd[i, neighbors] += args.epsilon / num_neighbors
            # targets_x_hd = targets_x_hd / targets_x_hd.sum(dim=1, keepdim=True)
            targets_x_hd = labels_x.detach()

            # For medium expert
            for i in range(batch_size):
                neighbors = medium_nearest_neighbors_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_md[i, neighbors] += args.epsilon / num_neighbors
            
            targets_x_md = targets_x_md / targets_x_md.sum(dim=1, keepdim=True)
            targets_x_md = targets_x_md.detach()

            # Track class distribution via soft labels
            for i in range(args.num_class):
                img_num_list_via_soft_label[i] += targets_x_tl.sum(dim=0)[i].cpu().numpy()

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        # Combine inputs and targets
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets_tl = torch.cat([targets_x_tl, targets_x_tl, targets_u, targets_u], dim=0)
        all_targets_hd = torch.cat([targets_x_hd, targets_x_hd, targets_u, targets_u], dim=0)
        all_targets_md = torch.cat([targets_x_md, targets_x_md, targets_u, targets_u], dim=0)

        # Create mixed samples
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a_tl, target_b_tl = all_targets_tl, all_targets_tl[idx]
        target_a_hd, target_b_hd = all_targets_hd, all_targets_hd[idx]
        target_a_md, target_b_md = all_targets_md, all_targets_md[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target_hd = l * target_a_hd + (1 - l) * target_b_hd
        mixed_target_tl = l * target_a_tl + (1 - l) * target_b_tl
        mixed_target_md = l * target_a_md + (1 - l) * target_b_md
        
        # Forward pass
        feats, logits = net(mixed_input, return_features=True)
        logits2 = net.classify2(feats)
        logits3 = net.classify3(feats)
        
        # Split predictions for labeled and unlabeled data
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        logits2_x = logits2[:batch_size*2]
        logits2_u = logits2[batch_size*2:]
        logits3_x = logits3[:batch_size*2]
        logits3_u = logits3[batch_size*2:]
        
        # Calculate losses
        Lx, Lu, lamb = criterion(logits_x, mixed_target_hd[:batch_size*2], 
                                logits_u, mixed_target_hd[batch_size*2:], 
                                epoch+batch_idx/num_iter, warm_up)
        
        loss_BCE_x, loss_BCE_u = balanced_softmax_loss_Semi(
            logits2_x, mixed_target_tl[:batch_size*2], 
            logits2_u, mixed_target_tl[batch_size*2:], 
            tmp_img_num_list)
        
        loss_BCE_x_md, loss_BCE_u_md = balanced_softmax_loss_Semi(
            logits3_x, mixed_target_md[:batch_size*2], 
            logits3_u, mixed_target_md[batch_size*2:], 
            tmp_img_num_list, tau=0.5)
        
        # Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # Total loss
        loss = Lx + lamb * Lu + loss_BCE_x + lamb * loss_BCE_u + loss_BCE_x_md + lamb * loss_BCE_u_md + penalty
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            sys.stdout.write('\r')
            sys.stdout.write(f'{args.dataset}:{args.imb_type}_{args.imb_factor}_{args.noise_mode}_{args.noise_ratio} | '
                            f'Epoch [{epoch:3d}/{args.num_epochs:3d}] Iter[{batch_idx+1:3d}/{num_iter:3d}]\t '
                            f'Labeled loss: {Lx.item():.2f}  Unlabeled loss: {Lu.item():.2f}')
            sys.stdout.flush()
            
    return img_num_list_via_soft_label

def test(epoch, net1, net2):
    """Evaluate model performance on test set"""
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
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # Get predictions from both networks
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)
            
            # Get predictions from all classifiers
            outputs1 = net1.classify2(feats1)
            outputs11 = net1.classify3(feats1)
            outputs2 = net2.classify2(feats2)
            outputs21 = net2.classify3(feats2)
            
            # Ensemble prediction
            outputs = outputs1 + outputs2 + outputs01 + outputs02 + outputs11 + outputs21
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            # Store predictions and targets for analysis
            total_preds.append(predicted)
            total_targets.append(targets)
            
            # Categorize by shot type
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

    # Calculate overall accuracy
    acc = 100. * correct / total

    # Concatenate all predictions and targets
    total_preds = torch.cat(total_preds)
    total_targets = torch.cat(total_targets)
    
    # Calculate per-class accuracy
    cls_acc = [
        round(100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / 
              max(1, (total_targets == i).sum().item()), 2)
        for i in range(args.num_class)
    ]
    
    # Calculate shot-specific accuracies
    def calculate_category_accuracy(preds, targets):
        if len(targets) == 0:
            return 0
        correct_preds = (torch.tensor(preds) == torch.tensor(targets)).sum().item()
        return 100. * correct_preds / len(targets)

    many_shot_acc = calculate_category_accuracy(many_shot_preds, many_shot_targets)
    medium_shot_acc = calculate_category_accuracy(medium_shot_preds, medium_shot_targets)
    few_shot_acc = calculate_category_accuracy(few_shot_preds, few_shot_targets)

    # Return comprehensive results
    return {
        "overall_accuracy": acc,
        "class_accuracy": cls_acc,
        "many_shot_accuracy": many_shot_acc,
        "medium_shot_accuracy": medium_shot_acc,
        "few_shot_accuracy": few_shot_acc
    }

def eval_train(model, cfeats_EMA, cfeats_sq_EMA):
    """Evaluate training data to identify clean samples"""
    model.eval()
    total_features = torch.zeros((num_all_img, feat_size))  # Sample features
    total_labels = torch.zeros(num_all_img).long()  # Sample labels
    tmp_img_num_list = torch.zeros(args.num_class)  # Clean samples per class
    pred = np.zeros(num_all_img, dtype=bool)  # Clean probability
    confs_BS = torch.zeros(num_all_img)  # Confidence from classifier
    confs_BS_max = torch.zeros(num_all_img)  # Max confidence
    mask = torch.zeros(num_all_img).bool()

    # Extract features and confidences
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, outputs = model(inputs, return_features=True)
            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            
            # Store confidences and features
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
            confs_BS_max[index] = probs2.max(dim=1)[0].cpu()
            
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]

    total_features = total_features.cuda()
    total_labels = total_labels.cuda()

    # Select high-confidence samples
    tau = 1 / args.num_class
    sampled_class = []  # High confidence classes
    for i in range(args.num_class):
        idx_selected = (confs_BS[idx_class[i]] > tau * args.phi ** epoch).nonzero(as_tuple=True)[0]
        idx_selected = idx_class[i][idx_selected]
        mask[idx_selected] = True
        if idx_selected.size(0) > 300:
            sampled_class.append(i)
        
    remained_class = list(set(range(args.num_class)) - set(sampled_class))

    # Update class feature EMA
    if epoch == args.warm_up + 1:  # First epoch, might select zero tail class samples
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=None)
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)
        # Handle cases where no samples were selected for a class
        refined_cfeats['cl2ncs'] = np.nan_to_num(refined_cfeats['cl2ncs'], nan=0.0)
        cfeats_EMA = args.beta * cfeats_EMA + (1 - args.beta) * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = args.beta * cfeats_sq_EMA + (1 - args.beta) * refined_cfeats['cl2ncs'] ** 2
    
    # Find k nearest neighbor prototypes
    def find_knn_with_faiss_cosine(features, prototypes, k):
        # Normalize features for cosine similarity
        features = features - features.mean(dim=1, keepdim=True)
        norm_features = torch.norm(features, p=2, dim=1, keepdim=True)
        normalized_features = features / norm_features
        normalized_features_np = normalized_features.cpu().numpy().astype(np.float32)

        # Create FAISS index for fast similarity search
        index = faiss.IndexFlatIP(prototypes.shape[1])
        index.add(prototypes)
        
        # Search for nearest neighbors
        _, indices = index.search(normalized_features_np, k)
        return normalized_features_np, indices

    # Find nearest neighbors for each sample
    k = int(args.num_class * args.k)
    normalized_features_np, nearest_neighbors_idx = find_knn_with_faiss_cosine(total_features, cfeats_EMA, k)
    
    # Categorize neighbors by shot type
    head_nearest_neighbors_idx = []
    medium_nearest_neighbors_idx = []
    tail_nearest_neighbors_idx = []

    for i in range(nearest_neighbors_idx.shape[0]):
        current_neighbors = nearest_neighbors_idx[i]
        
        # Filter by shot category
        head_neighbors = [label for label in current_neighbors if label in many_shot_classes]
        medium_neighbors = [label for label in current_neighbors if label in medium_shot_classes]
        tail_neighbors = [label for label in current_neighbors if label in few_shot_classes]
        
        head_nearest_neighbors_idx.append(head_neighbors[:k])
        medium_nearest_neighbors_idx.append(medium_neighbors[:k])
        tail_nearest_neighbors_idx.append(tail_neighbors[:k])

    # Convert to numpy arrays
    head_nearest_neighbors_idx = np.array(head_nearest_neighbors_idx, dtype=object)
    medium_nearest_neighbors_idx = np.array(medium_nearest_neighbors_idx, dtype=object)
    tail_nearest_neighbors_idx = np.array(tail_nearest_neighbors_idx, dtype=object)

    # Save nearest neighbors data
    save_dir = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}'
    os.makedirs(save_dir, exist_ok=True)
    
    for neighbor_type, neighbors in [
        ('head', head_nearest_neighbors_idx), 
        ('medium', medium_nearest_neighbors_idx),
        ('tail', tail_nearest_neighbors_idx)
    ]:
        df = pd.DataFrame(neighbors)
        save_path = f'{save_dir}/knn_{neighbor_type}_{epoch}.csv'
        df.to_csv(save_path, index=False)
    
    # Compute silhouette score to evaluate cluster quality
    silhouette_avg = silhouette_score(normalized_features_np, noisy_labels)
    print(f"silhouette score: {silhouette_avg:.6f}")
    
    # Sample centers from Gaussian posterior for robust estimation
    refined_ncm_logits = torch.zeros((num_all_img, args.num_class)).cuda()
    for i in range(args.sample_rate):
        mean = cfeats_EMA
        std = np.sqrt(np.clip(cfeats_sq_EMA - mean ** 2, 1e-30, 1e30))
        eps = np.random.normal(size=mean.shape)
        cfeats = mean + std * eps

        # Use original means for low-confidence classes
        refined_cfeats['cl2ncs'][sampled_class] = cfeats[sampled_class]
        refined_cfeats['cl2ncs'][remained_class] = mean[remained_class]

        # Get logits from nearest centroid classifier
        ncm_classifier.update(refined_cfeats, device=args.gpuid)
        refined_ncm_logits += ncm_classifier(total_features, None)[0]

    # Calculate probability of each sample being clean
    prob = get_gmm_prob(refined_ncm_logits, total_labels)
    
    # Select clean samples based on probability threshold
    for i in range(args.num_class):
        pred[idx_class[i]] = (prob[idx_class[i]] > args.p_threshold)
        tmp_img_num_list[i] = np.sum(pred[idx_class[i]])

    print('Number of clean samples selected per class: ', tmp_img_num_list)
    
    # Calculate selection precision and recall
    correct_predictions = ((torch.from_numpy(pred) == (noisy_labels == clean_labels)) & torch.from_numpy(pred)).sum()
    precision = correct_predictions / torch.from_numpy(pred).sum()
    
    # Calculate per-class precision
    cls_prec = [
        round(100. * ((torch.from_numpy(pred) == (clean_labels == noisy_labels)) & 
                     (clean_labels == i) & torch.from_numpy(pred)).sum().item() / 
              tmp_img_num_list[i].item(), 2) 
        if tmp_img_num_list[i].item() != 0 else 'nan'
        for i in range(args.num_class)
    ]
    
    # Calculate overall and per-class recall
    actual_clean = (noisy_labels == clean_labels).sum()
    recall = correct_predictions / actual_clean
    
    cls_recall = [
        round(100. * ((torch.from_numpy(pred) == (clean_labels == noisy_labels)) & 
                     (clean_labels == i) & torch.from_numpy(pred)).sum().item() /
              max(1, ((clean_labels == i) & (noisy_labels == clean_labels)).sum().item()), 2) 
        if ((clean_labels == i) & (noisy_labels == clean_labels)).sum().item() != 0 else 'nan'
        for i in range(args.num_class)
    ]
    
    print(f"Precision: {precision.item():.4f}% {str(cls_prec)}\nRecall: {recall.item():.4f}% {str(cls_recall)}")
    
    # Feature visualization with t-SNE (periodically)
    if epoch % 50 == 0:
        tsne = TSNE(n_components=2, random_state=0)
        features_2d = tsne.fit_transform(normalized_features_np)
        np.save(f'{save_dir}/featur_2d_{epoch}.npy', features_2d)

    return pred, confs_BS_max, tmp_img_num_list, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx, silhouette_avg

def get_gmm_prob(ncm_logits, total_labels):
    """Calculate probability of clean samples using GMM"""
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        # Get logits for this class
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # Normalize logits to [0, 1]
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        # Fit GMM with 2 components (clean and noisy)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        
        # Take probabilities from component with higher mean (clean component)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()

def get_knncentroids(feats=None, labels=None, mask=None):
    """Compute KNN centroids from features"""
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

    # Get unnormalized centroids
    un_centers = get_centroids(feats, labels, mask)
    
    # Get L2-normalized centroids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get centrally L2-normalized centroids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {
        'mean': featmean,
        'uncs': un_centers,
        'l2ncs': l2n_centers,
        'cl2ncs': cl2n_centers
    }

def feature_extractor(model, eval_loader):
    """Extract features from all samples using the model"""
    model.eval()
    total_features = torch.zeros((num_all_img, feat_size))

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            feats, _ = model(inputs, return_features=True)
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
    return total_features.numpy()

def linear_rampup(current, warm_up, rampup_length=16):
    """Linear ramp-up function for consistency loss weight"""
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)

class SemiLoss(object):
    """Semi-supervised learning loss combining supervised and unsupervised losses"""
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch, warm_up)

def create_model():
    """Create model with multiple classification heads"""
    model = PreActResNet18_3(num_classes=args.num_class)
    model = model.cuda()
    return model

def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1.0):
    """Balanced softmax loss accounting for class imbalance"""
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

def balanced_softmax_loss_Semi(logits_x, targets_x, logits_u, targets_u, sample_per_class, tau=1.0):
    """Balanced softmax loss for semi-supervised learning"""
    spc = sample_per_class.type_as(logits_x)

    # For labeled samples
    spc_x = spc.unsqueeze(0).expand(logits_x.shape[0], -1)
    logits_x = logits_x + tau * spc_x.log()
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    # For unlabeled samples
    spc_u = spc.unsqueeze(0).expand(logits_u.shape[0], -1)
    logits_u = logits_u + tau * spc_u.log()
    probs_u = torch.softmax(logits_u, dim=1)
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu

# Set warm-up period
warm_up = args.warm_up if args.warm_up is not None else 30

# Create checkpoint directory
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')

# Load data
loader = dataloader.cifar_dataloader(
    args.dataset, 
    imb_type=args.imb_type, 
    imb_factor=args.imb_factor, 
    noise_mode=args.noise_mode, 
    noise_ratio=args.noise_ratio,
    batch_size=args.batch_size, 
    num_workers=5, 
    root_dir=args.data_path
)

# Set number of classes
args.num_class = 100 if args.dataset == 'cifar100' else 10
feat_size = 512

# Initialize KNN classifier
ncm_classifier = KNNClassifier(feat_size, args.num_class)

# Get labels
clean_labels = torch.Tensor(loader.run('warmup').dataset.clean_targets).long()
noisy_labels = torch.Tensor(loader.run('warmup').dataset.noise_label).long()
real_img_num_list = torch.Tensor(loader.run('warmup').dataset.real_img_num_list).long()

# Save labels for analysis
save_dir = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}'
os.makedirs(save_dir, exist_ok=True)

for name, data in [
    ('noisy_labels', noisy_labels),
    ('clean_labels', clean_labels),
    ('real_img_num_list', real_img_num_list)
]:
    pd.DataFrame(data).to_csv(f'{save_dir}/{name}.csv', index=False)

# Load dataloaders
warmup_trainloader = loader.run('warmup')
eval_loader = loader.run('eval_train')
num_all_img = len(warmup_trainloader.dataset)

# Get indices for each class
idx_class = []
for i in range(args.num_class):
    idx_class.append((torch.tensor(warmup_trainloader.dataset.noise_label) == i).nonzero(as_tuple=True)[0])

# Compute class statistics
class_counts = [0] * args.num_class
for _, targets, _ in warmup_trainloader:
    for target in targets:
        class_counts[target.item()] += 1

# Sort classes by frequency
sorted_class_counts = sorted(enumerate(class_counts), key=lambda x: x[1], reverse=True)
sorted_classes = [class_idx for class_idx, _ in sorted_class_counts]

# Define shot categories
total_classes = len(class_counts)
top_30_percent = int(total_classes * 0.3)
bottom_30_percent = int(total_classes * 0.3)

many_shot_classes = set(sorted_classes[:top_30_percent])
few_shot_classes = set(sorted_classes[-bottom_30_percent:])
medium_shot_classes = set(sorted_classes[top_30_percent:-bottom_30_percent])

# Initialize feature storage
cfeats_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA1 = np.zeros((args.num_class, feat_size))
cfeats_EMA2 = np.zeros((args.num_class, feat_size))
cfeats_sq_EMA2 = np.zeros((args.num_class, feat_size))

# Build networks
print('| Building networks')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

# Load pre-trained weights
if os.path.isfile(args.pretrained):
    print(f"=> Loading checkpoint '{args.pretrained}'")
    checkpoint = torch.load(args.pretrained, map_location="cpu")

    # Rename MoCo pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
            state_dict[k[len("encoder_q."):]] = state_dict[k]
        del state_dict[k]

    # Load weights
    net1.load_state_dict(state_dict, strict=False)
    net2.load_state_dict(state_dict, strict=False)
    
    # Check for expected missing keys
    expected_missing = {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias", 
                       "linear3.weight", "linear3.bias"}
    assert set(net1.load_state_dict(state_dict, strict=False).missing_keys) == expected_missing
    assert set(net2.load_state_dict(state_dict, strict=False).missing_keys) == expected_missing
    
    print(f"=> Loaded pre-trained model '{args.pretrained}'")
else:
    print(f"=> No checkpoint found at '{args.pretrained}'")

# Create loss and optimizers
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Resume from checkpoint if specified
if args.resume is not None:
    resume_path = f'./checkpoint/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{args.resume}.pth'
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)
    print(f'| Loading model from {resume_path}')
    
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        prob1 = ckpt['prob1'] if 'prob1' in ckpt else None
        prob2 = ckpt['prob2'] if 'prob2' in ckpt else None
        
        # Load nearest neighbor indices
        head_nearest_neighbors_idx1 = ckpt['head_nearest_neighbors_idx1'] if 'head_nearest_neighbors_idx1' in ckpt else None
        medium_nearest_neighbors_idx1 = ckpt['medium_nearest_neighbors_idx1'] if 'medium_nearest_neighbors_idx1' in ckpt else None
        tail_nearest_neighbors_idx1 = ckpt['tail_nearest_neighbors_idx1'] if 'tail_nearest_neighbors_idx1' in ckpt else None
        head_nearest_neighbors_idx2 = ckpt['head_nearest_neighbors_idx2'] if 'head_nearest_neighbors_idx2' in ckpt else None
        medium_nearest_neighbors_idx2 = ckpt['medium_nearest_neighbors_idx2'] if 'medium_nearest_neighbors_idx2' in ckpt else None
        tail_nearest_neighbors_idx2 = ckpt['tail_nearest_neighbors_idx2'] if 'tail_nearest_neighbors_idx2' in ckpt else None
        
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume. Starting from scratch.')
        start_epoch = 1

# Initialize loss functions
CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()

# Training loop
acc_list = []
best_acc = 0
best_epoch = 0
best_model = None

for epoch in range(start_epoch, args.num_epochs + 1):
    # Adjust learning rate
    lr = args.lr
    if epoch > 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    
    # Load data for this epoch
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')
    
    # Warm-up phase
    if epoch <= warm_up:
        warmup_trainloader = loader.run('warmup')
        
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    
    # Main training phase with co-teaching
    else:
        current_labels1 = noisy_labels
        current_labels2 = noisy_labels
        
        # Evaluate training data to select clean samples
        pred1, prob1, tmp_img_num_list1, head_nearest_neighbors_idx1, medium_nearest_neighbors_idx1, tail_nearest_neighbors_idx1, silhouette_avg1 = eval_train(net1, cfeats_EMA1, cfeats_sq_EMA1)
        pred2, prob2, tmp_img_num_list2, head_nearest_neighbors_idx2, medium_nearest_neighbors_idx2, tail_nearest_neighbors_idx2, silhouette_avg2 = eval_train(net2, cfeats_EMA2, cfeats_sq_EMA2)
        
        # Train Net1 with samples selected by Net2
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, 
              current_labels2, tmp_img_num_list2, 
              head_nearest_neighbors_idx1, medium_nearest_neighbors_idx1, tail_nearest_neighbors_idx1)
        
        # Train Net2 with samples selected by Net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, 
              current_labels1, tmp_img_num_list1,
              head_nearest_neighbors_idx2, medium_nearest_neighbors_idx2, tail_nearest_neighbors_idx2)
    
    # Test models
    acc_dict = test(epoch, net1, net2)
    test_acc = acc_dict['overall_accuracy']
    acc_list.append(acc_dict)
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_model = {
            'net1': net1.state_dict(),
            'net2': net2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'acc_dict': acc_dict,
            # Add nearest neighbor indices
            'head_nearest_neighbors_idx1': head_nearest_neighbors_idx1,
            'medium_nearest_neighbors_idx1': medium_nearest_neighbors_idx1,
            'tail_nearest_neighbors_idx1': tail_nearest_neighbors_idx1,
            'head_nearest_neighbors_idx2': head_nearest_neighbors_idx2,
            'medium_nearest_neighbors_idx2': medium_nearest_neighbors_idx2,
            'tail_nearest_neighbors_idx2': tail_nearest_neighbors_idx2
        }
    
    # Save predictions and probabilities
    if epoch > warm_up:
        save_dir = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}'
        os.makedirs(save_dir, exist_ok=True)
        
        pd.DataFrame(pred2).to_csv(f'{save_dir}/prediction_{epoch}.csv', index=False)
        pd.DataFrame(prob2).to_csv(f'{save_dir}/probability_{epoch}.csv', index=False)
    
    # Print results
    print(f"\n| Test Epoch #{epoch}\t Accuracy: {test_acc:.2f}%\t Best Acc: {best_acc:.2f}%\t at Epoch {best_epoch}.")
    print(f"| Class Accuracies: {acc_dict['class_accuracy']}")
    print(f"| Many Shot Accuracy: {acc_dict['many_shot_accuracy']:.2f}%")
    print(f"| Medium Shot Accuracy: {acc_dict['medium_shot_accuracy']:.2f}%")
    print(f"| Few Shot Accuracy: {acc_dict['few_shot_accuracy']:.2f}%")
    
    # Save checkpoint
    if epoch in [warm_up, args.num_epochs] or epoch % 50 == 0:
        ckpt_dir = f'./checkpoint/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{epoch}.pth'
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        print(f'| Saving model to {ckpt_dir}')
        
        ckpt = {
            'net1': net1.state_dict(),
            'net2': net2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'prob1': prob1 if 'prob1' in locals() else None,
            'prob2': prob2 if 'prob2' in locals() else None,
            # Add nearest neighbor indices
            'head_nearest_neighbors_idx1': head_nearest_neighbors_idx1 if 'head_nearest_neighbors_idx1' in locals() else None,
            'medium_nearest_neighbors_idx1': medium_nearest_neighbors_idx1 if 'medium_nearest_neighbors_idx1' in locals() else None,
            'tail_nearest_neighbors_idx1': tail_nearest_neighbors_idx1 if 'tail_nearest_neighbors_idx1' in locals() else None,
            'head_nearest_neighbors_idx2': head_nearest_neighbors_idx2 if 'head_nearest_neighbors_idx2' in locals() else None,
            'medium_nearest_neighbors_idx2': medium_nearest_neighbors_idx2 if 'medium_nearest_neighbors_idx2' in locals() else None,
            'tail_nearest_neighbors_idx2': tail_nearest_neighbors_idx2 if 'tail_nearest_neighbors_idx2' in locals() else None
        }
        torch.save(ckpt, ckpt_dir)

# Save final accuracy results
save_dir = f'./saved/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}'
os.makedirs(save_dir, exist_ok=True)
pd.DataFrame(acc_list).to_csv(f'{save_dir}/Acc.csv', index=True)

# Save the best model
if best_model is not None:
    best_model_path = f'./checkpoint/{file_name}/{args.dataset}_{args.noise_ratio}_{args.imb_factor}_{args.epsilon}_{args.k}_best_model.pth'
    print(f"| Saving the best model at epoch {best_epoch} with accuracy {best_acc:.2f}%")
    torch.save(best_model, best_model_path)