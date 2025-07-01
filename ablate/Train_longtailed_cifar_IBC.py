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
from utils.resnet import resnet32
import utils.dataloader_longtailed_cifar as dataloader  # Import the new dataloader
import matplotlib.pyplot as plt
import faiss
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training for Long-tailed Distribution')
# training hyperparameters
parser.add_argument('-w', '--warm_up', default=30, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--gpuid', default=0, type=int, required=True)
parser.add_argument('--seed', default=123)

# model hyperparameters
parser.add_argument('--arch', default='PreActResNet18', type=str, help='model architechture')
parser.add_argument('-r', '--resume', default=None, type=int)

# dataset hyperparameters
parser.add_argument('--data_path', default='/nas/datasets', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--imb_type', default='exp', type=str)
parser.add_argument('--imb_factor', default=0.1, type=float)

# method hyperparameters
parser.add_argument('--k', default=0.5, type=float)
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--alpha', default=4, type=float, help='parameter for mixmatch')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--beta', default=0.99, type=float, help='smoothing factor')
parser.add_argument('--phi', default=1.005, type=float, help='parameter for dynamic threshold')
parser.add_argument('--sample_rate', default=5, type=int, help='sampling rate of SFA')

file_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
args.pretrained=f"../moco_ckpt/{args.arch}/{args.dataset}_exp_{args.imb_factor}/checkpoint_2000.pth.tar"

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
        loss_BCE = balanced_softmax_loss(labels, logits2, torch.tensor(cls_num_list), "mean", tau=0.5)

        logits3 = net.classify3(feats)
        loss_BCE_3 = balanced_softmax_loss(labels, logits3, torch.tensor(cls_num_list), "mean")

        L = loss + loss_BCE + loss_BCE_3
        L.backward()
        optimizer.step()

        if batch_idx%50==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(args.dataset, args.imb_type, args.imb_factor, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()

# Training - simplified without noise handling components
def train(epoch, net, optimizer, trainloader, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx):
    net.train()
    num_iter = (len(trainloader.dataset)//args.batch_size)+1
    
    for batch_idx, (inputs, labels, index) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        batch_size = inputs.size(0)
        
        # Create one-hot encoding for labels
        labels_one_hot = torch.zeros(batch_size, args.num_class, device=labels.device).scatter_(1, labels.view(-1,1), 1)
        
        # Feature extraction
        feats, logits = net(inputs, return_features=True)
        
        # Get outputs from different classifiers (for head, medium, tail)
        logits2 = net.classify2(feats)  # Tail expert
        logits3 = net.classify3(feats)  # Medium expert
        
        # Create refined targets for different experts
        targets_tl = torch.zeros_like(labels_one_hot, dtype=float)
        targets_hd = torch.zeros_like(labels_one_hot, dtype=float)
        targets_md = torch.zeros_like(labels_one_hot, dtype=float)
        
        # Base confidence for the true label
        targets_tl[range(batch_size), labels] = 1 - args.epsilon
        targets_hd[range(batch_size), labels] = 1 - args.epsilon
        targets_md[range(batch_size), labels] = 1 - args.epsilon
        
        # Increase confidence for nearest neighbors (class prototypes)
        for i in range(batch_size):
            # For tail expert
            tail_neighbors = tail_nearest_neighbors_idx[index[i]]
            num_tail_neighbors = len(tail_neighbors)
            if num_tail_neighbors > 0:
                targets_tl[i, tail_neighbors] += args.epsilon / num_tail_neighbors
            
            # For head expert - using original label
            targets_hd = labels_one_hot.detach()
            
            # For medium expert
            medium_neighbors = medium_nearest_neighbors_idx[index[i]]
            num_medium_neighbors = len(medium_neighbors)
            if num_medium_neighbors > 0:
                targets_md[i, medium_neighbors] += args.epsilon / num_medium_neighbors
        
        # Normalize targets to sum to 1
        targets_tl = targets_tl / targets_tl.sum(dim=1, keepdim=True)
        targets_md = targets_md / targets_md.sum(dim=1, keepdim=True)
        
        # Calculate losses for different classifiers
        loss_main = CEloss(logits, labels)
        spc = torch.tensor(cls_num_list).cuda()
        adjusted_logits_medium = logits2 + 0.5 * spc.log()
        adjusted_logits_tail = logits3 + 1 * spc.log()
        loss_BCE_medium = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_medium, dim=1) * targets_md, dim=1))
        loss_BCE_tail = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_tail, dim=1) * targets_tl, dim=1))
        
        # Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        
        # Total loss
        loss = loss_main + loss_BCE_tail + loss_BCE_medium + penalty
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%50==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.2f'
                    %(args.dataset, args.imb_type, args.imb_factor, epoch, args.num_epochs, batch_idx+1, num_iter, loss_main.item()))
            sys.stdout.flush()


def test(epoch, net):
    net.eval()
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
            feats, outputs = net(inputs, return_features=True)           
            outputs_tail = net.classify2(feats)
            outputs_medium = net.classify3(feats)
            outputs = outputs + outputs_tail + outputs_medium  # Ensemble of classifiers

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
    
    cls_acc = [round(100. * ((total_preds == total_targets) & (total_targets == i)).sum().item() / max(1, (total_targets == i).sum().item()), 2) 
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


# Feature extraction and nearest neighbors computation
def compute_features_and_neighbors(model):
    model.eval()
    total_features = torch.zeros((len(train_dataset), 64))  # 64 is feature dimension
    total_labels = torch.zeros(len(train_dataset)).long()

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            feats, _ = model(inputs, return_features=True)
            
            for i, idx in enumerate(index):
                total_features[idx] = feats[i]
                total_labels[idx] = targets[i]

    # Compute class centers
    class_features = []
    for c in range(args.num_class):
        idx = (total_labels == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            class_features.append(total_features[idx].mean(dim=0))
        else:
            class_features.append(torch.zeros(64).cuda())
    
    class_features = torch.stack(class_features)
    
    # Find k nearest neighbors for each sample
    k = int(args.num_class * args.k)
    total_features_np = total_features.cpu().numpy()
    class_features_np = class_features.cpu().numpy()
    
    # Normalize features for cosine similarity
    total_features_np = total_features_np / np.linalg.norm(total_features_np, axis=1, keepdims=True)
    class_features_np = class_features_np / np.linalg.norm(class_features_np, axis=1, keepdims=True)
    
    # Use FAISS for efficient nearest neighbors search
    index = faiss.IndexFlatIP(class_features_np.shape[1])
    index.add(class_features_np)
    _, nearest_neighbors_idx = index.search(total_features_np, k)
    
    # Generate specific nearest neighbors for head, medium, tail
    head_nearest_neighbors_idx = []
    medium_nearest_neighbors_idx = []
    tail_nearest_neighbors_idx = []

    for i in range(nearest_neighbors_idx.shape[0]):
        current_neighbors = nearest_neighbors_idx[i]
        
        # Filter neighbors by category
        head_neighbors = [label for label in current_neighbors if label in many_shot_classes]
        medium_neighbors = [label for label in current_neighbors if label in medium_shot_classes]
        tail_neighbors = [label for label in current_neighbors if label in few_shot_classes]

        head_nearest_neighbors_idx.append(head_neighbors[:k])
        medium_nearest_neighbors_idx.append(medium_neighbors[:k])
        tail_nearest_neighbors_idx.append(tail_neighbors[:k])
    
    return np.array(head_nearest_neighbors_idx, dtype=object), np.array(medium_nearest_neighbors_idx, dtype=object), np.array(tail_nearest_neighbors_idx, dtype=object)


def create_model():
    if args.arch == 'PreActResNet18':
        model = PreActResNet18_3(num_classes=args.num_class)
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=args.num_class)
    model = model.cuda()
    return model

def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1.0):
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


# Initialize the warm_up period
warm_up = args.warm_up if args.warm_up is not None else 30

# Create checkpoint directory
if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')

# Load the new long-tailed dataloader
loader = dataloader.cifar_longtailed_dataloader(
    args.dataset, 
    imb_type=args.imb_type, 
    imb_factor=args.imb_factor, 
    batch_size=args.batch_size, 
    num_workers=5, 
    root_dir=args.data_path
)

args.num_class = 100 if args.dataset == 'cifar100' else 10

# Get train and test loaders
train_loader, cls_num_list = loader.run('train')
train_dataset = train_loader.dataset
test_loader = loader.run('test')

# Identify many, medium, and few shot classes based on class distribution
sorted_class_counts = sorted(enumerate(cls_num_list), key=lambda x: x[1], reverse=True)
total_classes = len(cls_num_list)
top_30_percent = int(total_classes * 0.3)
bottom_30_percent = int(total_classes * 0.3)

sorted_classes = [class_idx for class_idx, _ in sorted_class_counts]
many_shot_classes = set(sorted_classes[:top_30_percent])
few_shot_classes = set(sorted_classes[-bottom_30_percent:])
medium_shot_classes = set(sorted_classes[top_30_percent: -bottom_30_percent])

# Create and initialize the model
net = create_model()
cudnn.benchmark = True

if args.arch == "resnet32":
    # Load MoCo pre-trained weights if available
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # Rename MoCo pre-trained keys
        state_dict = checkpoint['state_dict']
        print(state_dict.keys())
        for k in list(state_dict.keys()):
            # Retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # Remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # Delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = net.load_state_dict(state_dict, strict=False)
        print("Actual missing keys:", msg.missing_keys)
        assert set(msg.missing_keys) == {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias", "linear3.weight", "linear3.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))
else:
    # Load MoCo pre-trained weights if available
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # Rename MoCo pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # Retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # Remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # Delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = net.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias", "linear3.weight", "linear3.bias"}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

# Set up optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Resume from checkpoint if specified
if args.resume is not None:
    resume_path = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{args.resume}.pth'
    if os.path.exists(resume_path):
        print(f'| Loading model from {resume_path}')
        ckpt = torch.load(resume_path)
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = args.resume + 1
    else:
        print('| Failed to resume.')
        start_epoch = 1
else:
    start_epoch = 1

# Loss functions
CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()

# Training loop
acc_list = []
best_acc = 0
best_epoch = 0
best_model = None

# Compute initial nearest neighbors
head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx = compute_features_and_neighbors(net)

for epoch in range(start_epoch, args.num_epochs + 1):   
    # Learning rate schedule
    lr = args.lr
    if epoch > 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Train phase
    if epoch <= warm_up:
        print('Warmup Net')
        warmup(epoch, net, optimizer, train_loader)
    else:
        print('Train Net')
        train(epoch, net, optimizer, train_loader, head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx)
    
    # Test phase
    acc_dict = test(epoch, net)
    test_acc = acc_dict['overall_accuracy']
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch = epoch
        best_model = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc_dict': acc_dict
        }

    acc_list.append(acc_dict)
    
    # Print results
    print(f"\n| Test Epoch #{epoch}\t Accuracy: {test_acc:6.2f}%\t Best Acc: {best_acc:6.2f}\t at Epoch {best_epoch}.")
    print(acc_dict['class_accuracy'])
    print(f"Many Shot Accuracy: {acc_dict['many_shot_accuracy']:6.2f}%")
    print(f"Medium Shot Accuracy: {acc_dict['medium_shot_accuracy']:6.2f}%")
    print(f"Few Shot Accuracy: {acc_dict['few_shot_accuracy']:6.2f}%")

    # Save checkpoint periodically
    if epoch in [warm_up, args.num_epochs] or epoch % 50 == 0:
        save_name = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.epsilon}_{args.k}_ep{epoch}.pth'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        print(f'| Saving model to {save_name}')
        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, save_name)
    
    # Recompute nearest neighbors every 10 epochs
    if epoch % 10 == 0:
        head_nearest_neighbors_idx, medium_nearest_neighbors_idx, tail_nearest_neighbors_idx = compute_features_and_neighbors(net)

# Save final accuracy results
df = pd.DataFrame(acc_list)
save_name = f'./saved/{file_name}/{args.dataset}_{args.imb_factor}_{args.epsilon}_{args.k}/Acc.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=True)

# Save the best model
if best_model is not None:
    best_model_path = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.epsilon}_{args.k}_best_model.pth'
    print(f"Saving the best model at epoch {best_epoch} with accuracy {best_acc:.2f}%")
    torch.save(best_model, best_model_path)