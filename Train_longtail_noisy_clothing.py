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
from utils.InceptionResNetV2 import *
from utils.KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import utils.dataloader_clothing1m as dataloader
import torchnet
import torch.multiprocessing as mp
import faiss

# Parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
# Training parameters
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize')
parser.add_argument('--warm_up', default=1, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid1', default=0, type=int)
parser.add_argument('--gpuid2', default=1, type=int)

# Method parameters
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--k', default=0.1, type=float)
parser.add_argument('--epsilon', default=0.25, type=float)

# Dataset parameters
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--data_path', default='/nas/datasets', type=str, help='path to dataset')
parser.add_argument('--feat_size', default=1536, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cuda1 = torch.device('cuda')
cuda2 = torch.device('cuda')

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, 
          tmp_img_num_list, device, whichnet, head_nn_idx, medium_nn_idx, tail_nn_idx):
    """Train network using labeled and unlabeled data"""
    criterion = SemiLoss()
    
    net.train()
    net2.eval()  # Fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):
        # Get unlabeled batch
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
            
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)        
        w_x = w_x.view(-1, 1).type(torch.FloatTensor) 

        # Move data to GPU
        inputs_x, inputs_x2, labels_x, w_x, index_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), index_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            # Average and sharpen predictions
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
            # 1. Reduce confidence for potentially noisy label
            targets_x_tl[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_hd[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon
            targets_x_md[range(batch_size), labels_x.argmax(dim=1)] = 1 - args.epsilon

            # 2. Increase confidence of nearest class prototypes for each expert
            # For tail expert
            for i in range(batch_size): 
                neighbors = tail_nn_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_tl[i, neighbors] += args.epsilon / num_neighbors
            targets_x_tl = targets_x_tl / targets_x_tl.sum(dim=1, keepdim=True)
            targets_x_tl = targets_x_tl.detach()

            # For head expert (using original labels)
            targets_x_hd = labels_x.detach()

            # For medium expert
            for i in range(batch_size):
                neighbors = medium_nn_idx[index_x[i]]
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    targets_x_md[i, neighbors] += args.epsilon / num_neighbors
            targets_x_md = targets_x_md / targets_x_md.sum(dim=1, keepdim=True)
            targets_x_md = targets_x_md.detach()
        
        # MixMatch augmentation
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        # Combine inputs and targets
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets_tl = torch.cat([targets_x_tl, targets_x_tl, targets_u, targets_u], dim=0)
        all_targets_hd = torch.cat([targets_x_hd, targets_x_hd, targets_u, targets_u], dim=0)
        all_targets_md = torch.cat([targets_x_md, targets_x_md, targets_u, targets_u], dim=0)

        # Create mixed samples
        idx = torch.randperm(all_inputs.size(0))
        mixed_input = l * all_inputs + (1 - l) * all_inputs[idx]
        mixed_target_hd = l * all_targets_hd + (1 - l) * all_targets_hd[idx]
        mixed_target_tl = l * all_targets_tl + (1 - l) * all_targets_tl[idx]
        mixed_target_md = l * all_targets_md + (1 - l) * all_targets_md[idx]
                
        # Forward pass through model
        feats, logits = net(mixed_input, return_features=True)
        logits2 = net.classify2(feats)
        logits3 = net.classify3(feats)
        
        # Calculate losses for different experts
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target_hd, dim=1))
        loss_BCE = balanced_softmax_loss(mixed_target_md, logits2, tmp_img_num_list, "mean", tau=1)
        loss_BCE_2 = balanced_softmax_loss(mixed_target_tl, logits3, tmp_img_num_list, "mean", tau=2)
        
        # Regularization (encourage uniform predictions)
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        # Total loss
        loss = Lx + loss_BCE + loss_BCE_2 + penalty
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if batch_idx % 500 == 0:
            sys.stdout.write('\n')
            sys.stdout.write(f' |{whichnet} Epoch [{epoch:3d}/{args.num_epochs:3d}] '
                            f'Iter[{batch_idx+1:4d}/{num_iter:4d}]\t Labeled loss: {Lx.item():.2f}')
            sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader, real_img_num_list, device, whichnet):
    """Initial training with all samples"""
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        # Forward pass
        feats, outputs = net(inputs, return_features=True)
        outputs2 = net.classify2(feats)
        outputs3 = net.classify3(feats)

        # Convert labels to one-hot
        labels_onehot = torch.zeros(labels.size(0), args.num_class).cuda().scatter_(1, labels.view(-1,1), 1)
        
        # Calculate losses
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * labels_onehot, dim=1))
        loss_BCE = balanced_softmax_loss(labels_onehot, outputs2, real_img_num_list, "mean")
        loss_BCE_2 = balanced_softmax_loss(labels_onehot, outputs3, real_img_num_list, "mean", tau=2)
        L = loss + loss_BCE + loss_BCE_2

        # Update weights
        L.backward()
        optimizer.step()

        # Logging
        if batch_idx % 500 == 0:
            sys.stdout.write('\n')
            sys.stdout.write(f'|{whichnet} Epoch [{epoch:3d}/{args.num_epochs:3d}] '
                            f'Iter[{batch_idx+1:4d}/{num_iter:4d}]\t '
                            f'Total-loss: {L.item():.4f} CE-loss: {loss.item():.4f} BCE-loss: {loss_BCE.item():.4f}')
            sys.stdout.flush()

        
def test(epoch, net1, net2, test_loader, device):
    """Evaluate model performance on test set"""
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    acc_meter.reset()
    
    net1.eval()
    net2.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            # Get predictions from both networks
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)
            
            # Get predictions from additional classifiers
            outputs1 = net1.classify2(feats1)
            outputs2 = net2.classify2(feats2)

            # Ensemble predictions
            outputs = outputs1 + outputs2 + outputs01 + outputs02
            acc_meter.add(outputs, targets)
            
    return acc_meter.value()

def eval_train(epoch, eval_loader, model, num_all_img, cfeats_EMA, cfeats_sq_EMA, device):
    """Evaluate training data for selecting clean samples and computing nearest neighbors"""
    model.eval()
    total_features = torch.zeros((num_all_img, args.feat_size))
    total_labels = torch.zeros(num_all_img).long()
    confs_BS = torch.zeros(num_all_img)
    mask = torch.zeros(num_all_img).bool()

    # Extract features and confidences
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats, _ = model(inputs, return_features=True)
            
            # Store features and labels
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]

            # Compute confidence scores
            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
    
    # Move tensors to GPU
    total_features = total_features.cuda()
    total_labels = total_labels.cuda()

    # Select high-confidence samples
    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i).nonzero(as_tuple=True)[0].cpu().numpy()
        idx_selected = (confs_BS[this_cls_idxs] > 0.02 * 1.005**epoch).nonzero(as_tuple=True)[0]
        idx_selected = this_cls_idxs[idx_selected]
        mask[idx_selected] = True
    
    # Update class feature EMA
    if epoch <= args.warm_up + 1:  # First epoch
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=None)
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)
        refined_cfeats['cl2ncs'] = np.nan_to_num(refined_cfeats['cl2ncs'], nan=0.0)  # Handle empty classes
        cfeats_EMA = 0.9 * cfeats_EMA + 0.1 * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = 0.9 * cfeats_sq_EMA + 0.1 * refined_cfeats['cl2ncs'] ** 2

    # Find k nearest neighbor prototypes for each sample
    k = int(args.num_class * args.k)
    normalized_features_np, nearest_neighbors_idx = find_knn_with_faiss_cosine(total_features, cfeats_EMA, k)

    # Categorize neighbors by shot type
    head_neighbors = []
    medium_neighbors = []
    tail_neighbors = []

    for i in range(nearest_neighbors_idx.shape[0]):
        current_neighbors = nearest_neighbors_idx[i]
        head_list = [label for label in current_neighbors if label in many_shot_classes]
        medium_list = [label for label in current_neighbors if label in medium_shot_classes]
        tail_list = [label for label in current_neighbors if label in few_shot_classes]
        head_neighbors.append(head_list[:k])
        medium_neighbors.append(medium_list[:k])
        tail_neighbors.append(tail_list[:k])

    # Convert to numpy arrays
    head_neighbors = np.array(head_neighbors, dtype=object)
    medium_neighbors = np.array(medium_neighbors, dtype=object)
    tail_neighbors = np.array(tail_neighbors, dtype=object)
    
    # Compute sample probabilities with GMM
    sample_rate = 1
    refined_ncm_logits = torch.zeros((num_all_img, args.num_class)).cuda()
    ncm_classifier = KNNClassifier(args.feat_size, args.num_class)
    
    for i in range(sample_rate):
        mean = cfeats_EMA
        std = np.sqrt(np.clip(cfeats_sq_EMA - mean ** 2, 1e-30, 1e30))
        eps = np.random.normal(size=mean.shape)
        cfeats = mean + std * eps

        refined_cfeats['cl2ncs'] = cfeats
        ncm_classifier.update(refined_cfeats, device=device)
        refined_ncm_logits += ncm_classifier(total_features, None)[0]

    prob = get_gmm_prob(refined_ncm_logits, total_labels, device)
    return prob, head_neighbors, medium_neighbors, tail_neighbors


def find_knn_with_faiss_cosine(features, prototypes, k):
    """Find k-nearest neighbors using FAISS"""
    # Center and normalize features
    features = features - features.mean(dim=1, keepdim=True)
    norm_features = torch.norm(features, p=2, dim=1, keepdim=True)
    normalized_features = features / norm_features
    normalized_features_np = normalized_features.cpu().numpy().astype(np.float32)

    # Create FAISS index
    index = faiss.IndexFlatIP(prototypes.shape[1])  # Use inner product for cosine similarity
    index.add(prototypes)
    
    # Search for nearest neighbors
    _, indices = index.search(normalized_features_np, k)
    return normalized_features_np, indices


def get_gmm_prob(ncm_logits, total_labels, device):
    """Calculate probability of samples being clean using GMM"""
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # Normalize logits to [0, 1]
        this_cls_logits -= np.min(this_cls_logits)
        if np.max(this_cls_logits) != 0:
            this_cls_logits /= np.max(this_cls_logits)

        # Fit GMM with 2 components (clean and noisy)
        gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
        gmm_prob = gmm.predict_proba(this_cls_logits)
        
        # Use probabilities from component with higher mean (clean component)
        this_cls_prob = torch.Tensor(gmm_prob[:, gmm.means_.argmax()]).cuda()
        prob[this_cls_idxs] = this_cls_prob

    return prob.cpu().numpy()


def get_knncentroids(feats=None, labels=None, mask=None):
    """Compute class centroids from features"""
    feats = feats.cpu().numpy()
    labels = labels.cpu().numpy()
    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        """Compute class centroids with optional mask"""
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype('bool')
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()
            
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_==i) & mask_], axis=0))
        return np.stack(centroids)

    # Compute different types of centroids
    un_centers = get_centroids(feats, labels, mask)  # Unnormalized centroids
    
    # L2-normalized centroids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Centrally L2-normalized centroids
    cl2n_feats = torch.Tensor(feats.copy()) - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {
        'mean': featmean,
        'uncs': un_centers,
        'l2ncs': l2n_centers,   
        'cl2ncs': cl2n_centers
    }


def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1):
    """Balanced softmax loss that accounts for class imbalance"""
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1))
    return loss


def balanced_softmax_loss_Semi(logits_x, targets_x, logits_u, targets_u, sample_per_class, tau=1.0):
    """Balanced softmax loss for semi-supervised learning"""
    # For labeled samples
    spc = sample_per_class.type_as(logits_x)
    spc_x = spc.unsqueeze(0).expand(logits_x.shape[0], -1)
    logits_x = logits_x + tau * spc_x.log()
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    # For unlabeled samples
    spc_u = spc.unsqueeze(0).expand(logits_u.shape[0], -1)
    logits_u = logits_u + tau * spc_u.log()
    probs_u = torch.softmax(logits_u, dim=1)
    Lu = torch.mean((probs_u - targets_u)**2)

    return Lx, Lu


def linear_rampup(current, warm_up, rampup_length=16):
    """Linear ramp-up function for consistency loss weight"""
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    """Semi-supervised loss combining supervised and unsupervised components"""
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


def create_model(device):
    """Create and initialize model"""
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model


# Set up logging
stats_log = open('./checkpoint/model_stats.txt', 'w') 
test_log = open('./checkpoint/model_acc.txt', 'w')         

warm_up = 1

# Initialize data loader
loader = dataloader.clothing_dataloader(
    batch_size=args.batch_size,
    num_class=args.num_class,
    num_workers=4,
    root_dir=args.data_path
)

print('| Building networks')
# Create networks
net1 = create_model(cuda1)
net2 = create_model(cuda2)
net1_clone = create_model(cuda2)
net2_clone = create_model(cuda1)
cudnn.benchmark = True

# Create optimizers
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Resume from checkpoint if specified
if args.resume is not None:
    resume_path = f'./checkpoint/model_webvision_e{args.resume}.pth'
    print(f'| Loading model from {resume_path}')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        # Load model and optimizer states
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        
        # Load additional data
        prob1 = ckpt['prob1']
        prob2 = ckpt['prob2']
        head_nearest_neighbors_idx1 = ckpt['head_nearest_neighbors_idx1']
        medium_nearest_neighbors_idx1 = ckpt['medium_nearest_neighbors_idx1']
        tail_nearest_neighbors_idx1 = ckpt['tail_nearest_neighbors_idx1']
        head_nearest_neighbors_idx2 = ckpt['head_nearest_neighbors_idx2']
        medium_nearest_neighbors_idx2 = ckpt['medium_nearest_neighbors_idx2']
        tail_nearest_neighbors_idx2 = ckpt['tail_nearest_neighbors_idx2']
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        start_epoch = 1
else:
    start_epoch = 1

# Initialize data loaders
web_valloader = loader.run('test')
warmup_trainloader1 = loader.run('warmup')
warmup_trainloader2 = loader.run('warmup')

# Get class statistics
real_img_num_list = torch.tensor(warmup_trainloader1.dataset.real_img_num_list)
idx_class = warmup_trainloader1.dataset.idx_class
num_all_img = torch.sum(real_img_num_list)
print(f"Class imbalance: max={max(real_img_num_list)}, min={min(real_img_num_list)}, ratio={max(real_img_num_list) / min(real_img_num_list):.2f}")

# Categorize classes by frequency
class_counts = [0] * args.num_class
for _, targets, _ in warmup_trainloader1:
    for target in targets:
        class_counts[target.item()] += 1

# Sort classes by frequency
sorted_class_counts = sorted(enumerate(class_counts), key=lambda x: x[1], reverse=True)
sorted_classes = [class_idx for class_idx, _ in sorted_class_counts]

# Define shot categories
total_classes = len(class_counts)
top_30_percent = int(total_classes * 0.3)
bottom_30_percent = int(total_classes * 0.3)

# Categorize classes by frequency
many_shot_classes = set(sorted_classes[:top_30_percent])
few_shot_classes = set(sorted_classes[-bottom_30_percent:])
medium_shot_classes = set(sorted_classes[top_30_percent:-bottom_30_percent])

# Initialize feature tracking variables
cfeats_EMA1 = np.zeros((args.num_class, args.feat_size))
cfeats_sq_EMA1 = np.zeros((args.num_class, args.feat_size))
cfeats_EMA2 = np.zeros((args.num_class, args.feat_size))
cfeats_sq_EMA2 = np.zeros((args.num_class, args.feat_size))

# Main training loop
for epoch in range(start_epoch, args.num_epochs + 1):
    # Learning rate schedule
    lr = args.lr
    if epoch == 50:
        lr /= 10
        
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr              

    # Warmup phase
    if epoch <= warm_up:
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader1, real_img_num_list, cuda1, 'net1')
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader2, real_img_num_list, cuda2, 'net2')
    # Main training phase
    else:
        # Evaluate data to select clean samples and find nearest neighbors
        eval_loader1 = loader.run('eval_train')
        eval_loader2 = loader.run('eval_train')
        
        prob1, head_nn_idx1, medium_nn_idx1, tail_nn_idx1 = eval_train(
            epoch, eval_loader1, net1, num_all_img, cfeats_EMA1, cfeats_sq_EMA1, cuda1)
        prob2, head_nn_idx2, medium_nn_idx2, tail_nn_idx2 = eval_train(
            epoch, eval_loader2, net2, num_all_img, cfeats_EMA2, cfeats_sq_EMA2, cuda2)
        
        # Select clean samples for co-training
        tmp_img_num_list1 = torch.zeros(args.num_class)
        pred1 = np.zeros(num_all_img, dtype=bool)
        tmp_img_num_list2 = torch.zeros(args.num_class)
        pred2 = np.zeros(num_all_img, dtype=bool)

        for i in range(args.num_class):
            pred1[idx_class[i]] = (prob1[idx_class[i]] > args.p_threshold)
            tmp_img_num_list1[i] = np.sum(pred1[idx_class[i]])

            pred2[idx_class[i]] = (prob2[idx_class[i]] > args.p_threshold)
            tmp_img_num_list2[i] = np.sum(pred2[idx_class[i]])

        # Co-divide training data
        labeled_trainloader1, unlabeled_trainloader1 = loader.run('train', pred2, prob2)
        labeled_trainloader2, unlabeled_trainloader2 = loader.run('train', pred1, prob1)
        
        # Train networks
        train(epoch, net1, net2_clone, optimizer1, labeled_trainloader1, unlabeled_trainloader1, 
              tmp_img_num_list2, cuda1, 'net1', head_nn_idx2, medium_nn_idx2, tail_nn_idx2)
        train(epoch, net2, net1_clone, optimizer2, labeled_trainloader2, unlabeled_trainloader2, 
              tmp_img_num_list1, cuda2, 'net2', head_nn_idx1, medium_nn_idx1, tail_nn_idx1)

    # Synchronize clones
    net1_clone.load_state_dict(net1.state_dict())
    net2_clone.load_state_dict(net2.state_dict())
    
    # Evaluate models
    web_acc = test(epoch, net1, net2_clone, web_valloader, cuda1)
    
    # Log results
    print(f"\n| Test Epoch #{epoch}\t Clothing1M Acc: {web_acc[0]:.2f}% ({web_acc[1]:.2f}%)\n")
    test_log.write(f'Epoch:{epoch} \t Clothing1M Acc: {web_acc[0]:.2f}% ({web_acc[1]:.2f}%)\n')
    test_log.flush()
    
    # Save checkpoint
    if epoch == 1 or epoch % 50 == 0:
        save_path = f'./checkpoint/model_webvision_e{epoch}.pth'
        print(f'| Saving model to {save_path}')

        ckpt = {
            'net1': net1.state_dict(),
            'net2': net2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'prob1': prob1 if 'prob1' in locals() else None,
            'prob2': prob2 if 'prob2' in locals() else None,
            # Store nearest neighbor indices
            'head_nearest_neighbors_idx1': head_nn_idx1 if 'head_nn_idx1' in locals() else None,
            'medium_nearest_neighbors_idx1': medium_nn_idx1 if 'medium_nn_idx1' in locals() else None,
            'tail_nearest_neighbors_idx1': tail_nn_idx1 if 'tail_nn_idx1' in locals() else None,
            'head_nearest_neighbors_idx2': head_nn_idx2 if 'head_nn_idx2' in locals() else None,
            'medium_nearest_neighbors_idx2': medium_nn_idx2 if 'medium_nn_idx2' in locals() else None,
            'tail_nearest_neighbors_idx2': tail_nn_idx2 if 'tail_nn_idx2' in locals() else None
        }
        torch.save(ckpt, save_path)