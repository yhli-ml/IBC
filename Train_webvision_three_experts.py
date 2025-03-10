from __future__ import print_function
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import sys
import argparse
import numpy as np
from InceptionResNetV2 import *
from KNNClassifier import KNNClassifier
from sklearn.mixture import GaussianMixture
import dataloader_webvision as dataloader
import torchnet
import torch.multiprocessing as mp
import faiss

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize')
parser.add_argument('--warm_up', default=1, type=int)
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid1', default=0, type=int)
parser.add_argument('--gpuid2', default=1, type=int)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--imb_factor', default=1, type=float)
parser.add_argument('--data_path', default='/nas/datasets', type=str, help='path to dataset')
parser.add_argument('--feat_size', default=1536, type=int)
parser.add_argument('-r', '--resume', default=None, type=int)
parser.add_argument('--k', default=0.1, type=float)
parser.add_argument('--epsilon', default=0.25, type=float)
args = parser.parse_args()

random.seed(args.seed)
cuda1 = torch.device('cuda')
cuda2 = torch.device('cuda')

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, tmp_img_num_list, device,whichnet, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx):
    criterion = SemiLoss()
    
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, index_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)            
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x, index_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), index_x.cuda()
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

        # Lx, Lu, lamb = criterion(logits_x, mixed_target_hd[:batch_size*2], logits_u, mixed_target_hd[batch_size*2:], epoch+batch_idx/num_iter, args.warm_up)
        # loss_BCE_x, loss_BCE_u = balanced_softmax_loss_Semi(logits2_x, mixed_target_tl[:batch_size*2], logits2_u, mixed_target_tl[batch_size*2:], tmp_img_num_list)
        # loss_BCE_x_md, loss_BCE_u_md = balanced_softmax_loss_Semi(logits3_x, mixed_target_md[:batch_size*2], logits3_u, mixed_target_md[batch_size*2:], tmp_img_num_list, tau=0.5)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target_hd, dim=1))
        loss_BCE = balanced_softmax_loss(mixed_target_md, logits2, tmp_img_num_list, "mean", tau=1)
        loss_BCE_2 = balanced_softmax_loss(mixed_target_tl, logits3, tmp_img_num_list, "mean", tau=2)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + loss_BCE + loss_BCE_2 + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 500 == 0:
            sys.stdout.write('\n')
            sys.stdout.write(' |%s Epoch [%3d/%3d] Iter[%4d/%4d]\t Labeled loss: %.2f'
                    %(whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
            sys.stdout.flush()


def warmup(epoch,net,optimizer,dataloader,real_img_num_list,device,whichnet):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        feats, outputs = net(inputs, return_features=True)
        outputs2 = net.classify2(feats)
        outputs3 = net.classify3(feats)

        labels = torch.zeros(labels.size(0), args.num_class).cuda().scatter_(1, labels.view(-1,1), 1)
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * labels, dim=1))
        loss_BCE = balanced_softmax_loss(labels, outputs2, real_img_num_list, "mean")
        loss_BCE_2 = balanced_softmax_loss(labels, outputs3, real_img_num_list, "mean", tau=2)
        L = loss + loss_BCE + loss_BCE_2

        L.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            sys.stdout.write('\n')
            sys.stdout.write('|%s  Epoch [%3d/%3d] Iter[%4d/%4d]\t  Total-loss: %.4f CE-loss: %.4f  BCE-loss: %.4f'
                    %(whichnet, epoch, args.num_epochs, batch_idx+1, num_iter, L.item(), loss.item(), loss_BCE.item()))
            sys.stdout.flush()

        
def test(epoch,net1,net2,test_loader,device):
    acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)
    acc_meter.reset()
    net1.eval()
    net2.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats1, outputs01 = net1(inputs, return_features=True)
            feats2, outputs02 = net2(inputs, return_features=True)    
            outputs1 = net1.classify2(feats1)
            outputs2 = net2.classify2(feats2)

            outputs = outputs1 + outputs2 + outputs01 + outputs02
            _, predicted = torch.max(outputs, 1)
            acc_meter.add(outputs,targets)
    accs = acc_meter.value()
    return accs

def eval_train(epoch, eval_loader, model, num_all_img, cfeats_EMA, cfeats_sq_EMA, device):    
    model.eval()
    total_features = torch.zeros((num_all_img, args.feat_size))
    total_labels = torch.zeros(num_all_img).long()
    confs_BS = torch.zeros(num_all_img)
    mask = torch.zeros(num_all_img).bool()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats, outputs = model(inputs, return_features=True)
            for b in range(inputs.size(0)):
                total_features[index[b]] = feats[b]
                total_labels[index[b]] = targets[b]

            logits2 = model.classify2(feats)
            probs2 = F.softmax(logits2, dim=1)
            confs_BS[index] = probs2[range(probs2.size(0)), targets].cpu()
    
    total_features = total_features.cuda()
    total_labels = total_labels.cuda()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i).nonzero(as_tuple=True)[0].cpu().numpy()
        idx_selected = (confs_BS[this_cls_idxs] > 0.02 * 1.005**epoch).nonzero(as_tuple=True)[0]
        idx_selected = this_cls_idxs[idx_selected]
        mask[idx_selected] = True
    
    if epoch <= args.warm_up + 1: # first epoch, might select zero tail class samples
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=None)  # (10, 1536)
        cfeats_EMA = refined_cfeats['cl2ncs']
        cfeats_sq_EMA = refined_cfeats['cl2ncs'] ** 2
    else:
        refined_cfeats = get_knncentroids(feats=total_features, labels=total_labels, mask=mask)  # (10, 1536)
        refined_cfeats['cl2ncs'] = np.nan_to_num(refined_cfeats['cl2ncs'], nan=0.0) # in case of none selected (especially tail class)
        cfeats_EMA = 0.9 * cfeats_EMA + 0.1 * refined_cfeats['cl2ncs']
        cfeats_sq_EMA = 0.9 * cfeats_sq_EMA + 0.1 * refined_cfeats['cl2ncs'] ** 2

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
    
    # sample centers from gaussion postier
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
    return prob,head_nearest_neighbors_idx,medium_nearest_neighbors_idx,tail_nearest_neighbors_idx


def get_gmm_prob(ncm_logits, total_labels, device):
    prob = torch.zeros_like(total_labels).float()

    for i in range(args.num_class):
        this_cls_idxs = (total_labels == i)
        this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
        # normalization, note that the logits are all negative
        this_cls_logits -=  np.min(this_cls_logits)
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

def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1))
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

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def create_model(device):
    model = InceptionResNetV2(num_classes=args.num_class)
    model = model.cuda()
    return model


torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)    

stats_log=open('./checkpoint/model_stats.txt','w') 
test_log=open('./checkpoint/model_acc.txt','w')         

warm_up = 1

loader = dataloader.webvision_dataloader(batch_size=args.batch_size,num_class = args.num_class,num_workers=4,root_dir=args.data_path,imb_ratio=args.imb_factor)
print('| Building net')

net1 = create_model(cuda1)
net2 = create_model(cuda2)

net1_clone = create_model(cuda2)
net2_clone = create_model(cuda1)

cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

if args.resume is not None:        
    resume_path = f'./checkpoint/model_webvision_e{args.resume}.pth'
    print(f'| Loading model from {resume_path}')
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path)
        net1.load_state_dict(ckpt['net1'])
        net2.load_state_dict(ckpt['net2'])
        optimizer1.load_state_dict(ckpt['optimizer1'])
        optimizer2.load_state_dict(ckpt['optimizer2'])
        prob1 = ckpt['prob1']
        prob2 = ckpt['prob2']
        head_nearest_neighbors_idx1=ckpt['head_nearest_neighbors_idx1']
        medium_nearest_neighbors_idx1=ckpt['medium_nearest_neighbors_idx1']
        tail_nearest_neighbors_idx1=ckpt['tail_nearest_neighbors_idx1']
        head_nearest_neighbors_idx2=ckpt['head_nearest_neighbors_idx2']
        medium_nearest_neighbors_idx2=ckpt['medium_nearest_neighbors_idx2']
        tail_nearest_neighbors_idx2=ckpt['tail_nearest_neighbors_idx2']
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        start_epoch = 1
else:
    start_epoch = 1

web_valloader = loader.run('test')
imagenet_valloader = loader.run('imagenet')   
warmup_trainloader1 = loader.run('warmup')
warmup_trainloader2 = loader.run('warmup')
real_img_num_list = torch.tensor(warmup_trainloader1.dataset.real_img_num_list)
idx_class = warmup_trainloader1.dataset.idx_class
num_all_img = torch.sum(real_img_num_list)
print(max(real_img_num_list), min(real_img_num_list), max(real_img_num_list) / min(real_img_num_list))

#-----------------------------------------------------  
class_counts = [0] * args.num_class
for _, targets, index in warmup_trainloader1:
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

cfeats_EMA1 = np.zeros((args.num_class, args.feat_size))
cfeats_sq_EMA1 = np.zeros((args.num_class, args.feat_size))
cfeats_EMA2 = np.zeros((args.num_class, args.feat_size))
cfeats_sq_EMA2 = np.zeros((args.num_class, args.feat_size))

for epoch in range(start_epoch, args.num_epochs + 1):   
    lr=args.lr
    if epoch == 50:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr              

    if epoch <= warm_up:
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader1,real_img_num_list,cuda1,'net1')
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader2,real_img_num_list,cuda2,'net2')

    else:
        eval_loader1 = loader.run('eval_train')
        eval_loader2 = loader.run('eval_train')
        prob1,head_nearest_neighbors_idx1,medium_nearest_neighbors_idx1,tail_nearest_neighbors_idx1=eval_train(epoch, eval_loader1, net1, num_all_img, cfeats_EMA1, cfeats_sq_EMA1, cuda1)
        prob2,head_nearest_neighbors_idx2,medium_nearest_neighbors_idx2,tail_nearest_neighbors_idx2=eval_train(epoch, eval_loader2, net2, num_all_img, cfeats_EMA2, cfeats_sq_EMA2, cuda2)
        
        tmp_img_num_list1 = torch.zeros(args.num_class)
        pred1 = np.zeros(num_all_img, dtype=bool)
        tmp_img_num_list2 = torch.zeros(args.num_class)
        pred2 = np.zeros(num_all_img, dtype=bool)

        for i in range(args.num_class):
            pred1[idx_class[i]] = (prob1[idx_class[i]] > args.p_threshold)
            tmp_img_num_list1[i] = np.sum(pred1[idx_class[i]])

            pred2[idx_class[i]] = (prob2[idx_class[i]] > args.p_threshold)
            tmp_img_num_list2[i] = np.sum(pred2[idx_class[i]])

        labeled_trainloader1, unlabeled_trainloader1 = loader.run('train', pred2, prob2) # co-divide
        labeled_trainloader2, unlabeled_trainloader2 = loader.run('train', pred1, prob1) # co-divide
        
        train(epoch,net1,net2_clone,optimizer1,labeled_trainloader1, unlabeled_trainloader1,tmp_img_num_list2,cuda1,'net1',head_nearest_neighbors_idx1, medium_nearest_neighbors_idx1, tail_nearest_neighbors_idx1)               
        train(epoch,net2,net1_clone,optimizer2,labeled_trainloader2, unlabeled_trainloader2,tmp_img_num_list1,cuda2,'net2',head_nearest_neighbors_idx2, medium_nearest_neighbors_idx2, tail_nearest_neighbors_idx2)

    net1_clone.load_state_dict(net1.state_dict())
    net2_clone.load_state_dict(net2.state_dict())
    
    web_acc = test(epoch,net1,net2_clone,web_valloader,cuda1)              
    imagenet_acc = test(epoch,net1_clone,net2,imagenet_valloader,cuda2)     
    
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()
    
    if epoch % 50 == 0:
        save_path = f'./checkpoint/model_webvision_e{epoch}.pth'
        print(f'| Saving model to {save_path}')

        ckpt = {'net1': net1.state_dict(),
                'net2': net2.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'prob1': prob1 if 'prob1' in dir() else None,
                'prob2': prob2 if 'prob2' in dir() else None,
                'head_nearest_neighbors_idx1':head_nearest_neighbors_idx1, 
                'medium_nearest_neighbors_idx1':medium_nearest_neighbors_idx1, 
                'tail_nearest_neighbors_idx1':tail_nearest_neighbors_idx1,
                'head_nearest_neighbors_idx2':head_nearest_neighbors_idx2, 
                'medium_nearest_neighbors_idx2':medium_nearest_neighbors_idx2, 
                'tail_nearest_neighbors_idx2':tail_nearest_neighbors_idx2}
        torch.save(ckpt, save_path)
