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
from PreResNet import *
from KNNClassifier import KNNClassifier
import dataloader_longtailed_cifar as dataloader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import faiss
from sklearn.metrics import silhouette_score
from PIL import Image
import cv2
import torchvision.transforms as transforms
import pandas as pd
from resnet import resnet32

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training for Long-tailed Distribution')
# training hyperparameters
parser.add_argument('-w', '--warm_up', default=30, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--gpuid', default=0, type=int, required=True)
parser.add_argument('--seed', default=123)

# model hyperparameters
parser.add_argument('--arch', default='resnet32', type=str, help='model architechture')
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
parser.add_argument('--tau_tail', default=1.0, type=float, help='tau for tail expert in balanced softmax')
parser.add_argument('--tau_medium', default=0.5, type=float, help='tau for medium expert in balanced softmax')
parser.add_argument('--max_smoothing', default=0.2, type=float, help='maximum semantic smoothing value')
parser.add_argument('--min_smoothing', default=0.05, type=float, help='minimum semantic smoothing value')
parser.add_argument('--similarity_temp', default=0.5, type=float, help='temperature for similarity calculation')
parser.add_argument('--update_freq', default=1, type=int, help='update frequency for class prototypes')

file_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
args.pretrained=f"../moco_ckpt/{args.arch}/{args.dataset}_exp_{args.imb_factor}/checkpoint_2000.pth.tar"

print(args)

class SemanticLabelSmoothing:
    def __init__(self, num_classes, max_smoothing=0.2, min_smoothing=0.05, similarity_temp=0.5, update_freq=5):
        """
        Semantic Label Smoothing based on feature-space similarity.
        
        Args:
            num_classes: Number of classes in the dataset
            max_smoothing: Maximum probability mass to redistribute
            min_smoothing: Minimum probability mass to redistribute 
            similarity_temp: Temperature for similarity distribution (lower = sharper)
            update_freq: How often to update class prototypes (in epochs)
        """
        self.num_classes = num_classes
        self.max_smoothing = max_smoothing
        self.min_smoothing = min_smoothing
        self.temperature = similarity_temp
        self.update_freq = update_freq
        
        # Initialize class prototypes and similarity matrix
        self.class_prototypes = None
        self.similarity_matrix = torch.eye(num_classes).cuda()  # Start with identity matrix
        self.last_update_epoch = 0
        
        print(f"Initialized Semantic Label Smoothing with max_smoothing={max_smoothing}, "
              f"min_smoothing={min_smoothing}, similarity_temp={similarity_temp}")
    
    def update_class_prototypes(self, model, dataloader, current_epoch):
        """
        Update class prototypes based on current model embeddings.
        """
        if current_epoch - self.last_update_epoch < self.update_freq:
            return False  # Skip if not time to update yet
            
        print(f"Updating class prototypes for semantic label smoothing at epoch {current_epoch}...")
        model.eval()
        
        # Initialize accumulators for features and counts per class
        total_features = [[] for _ in range(self.num_classes)]
        
        with torch.no_grad():
            for inputs, targets, _ in dataloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                features, _ = model(inputs, return_features=True)
                
                # Normalize features for cosine similarity
                features = F.normalize(features, p=2, dim=1)
                
                # Group features by class
                for i, target in enumerate(targets):
                    total_features[target.item()].append(features[i].detach().cpu())
        
        # Compute prototype (mean) for each class
        class_prototypes = []
        for class_idx in range(self.num_classes):
            if total_features[class_idx]:
                class_prototype = torch.stack(total_features[class_idx]).mean(0)
                class_prototypes.append(F.normalize(class_prototype, p=2, dim=0))
            else:
                # Handle empty classes (shouldn't happen with CIFAR)
                class_prototypes.append(torch.zeros(features.shape[1]))
        
        self.class_prototypes = torch.stack(class_prototypes)
        
        # Compute pairwise cosine similarity between class prototypes
        similarity_matrix = torch.mm(self.class_prototypes, self.class_prototypes.t())
        
        # Apply temperature scaling and zero out self-similarity
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix.fill_diagonal_(0)  # Zero out self-similarity
        
        # Apply stable softmax
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]  # For numerical stability
        similarity_matrix = torch.exp(similarity_matrix)
        row_sums = similarity_matrix.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)  # Prevent division by zero
        similarity_matrix = similarity_matrix / row_sums
        
        self.similarity_matrix = similarity_matrix.cuda()
        self.last_update_epoch = current_epoch
        
        # Debugging info
        top_similarities = []
        for i in range(self.num_classes):
            sim_vals, sim_indices = torch.topk(similarity_matrix[i], 3)
            top_similarities.append((i, [(sim_indices[j].item(), sim_vals[j].item()) for j in range(len(sim_indices))]))
        
        print("Sample class similarities (class_id -> [(similar_class, similarity_score), ...]):")
        for i, sims in random.sample(top_similarities, min(5, len(top_similarities))):
            print(f"  Class {i}: {sims}")
            
        return True
    
    def __call__(self, labels, num_classes, class_types, smoothing=None):
        """
        Create semantic smooth labels based on class similarity.
        
        Args:
            labels: True labels (batch_size)
            num_classes: Number of classes
            class_types: Tensor indicating class type for each instance (0=head, 1=medium, 2=tail)
            smoothing: Override smoothing factor (optional)
            
        Returns:
            smooth_labels: Semantically smoothed targets (batch_size, num_classes)
        """
        batch_size = labels.size(0)
        
        # If smoothing is 0, return one-hot labels
        if smoothing == 0:
            one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            return one_hot
        
        # Use provided smoothing value or max smoothing
        current_smoothing = self.max_smoothing if smoothing is None else smoothing
        
        # Create standard one-hot labels
        one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # For each instance, get its similarity vector based on its class
        smooth_labels = torch.zeros_like(one_hot)
        for i, (label, class_type) in enumerate(zip(labels, class_types)):
            # Apply selective smoothing based on class type
            # 0=head, 1=medium, 2=tail
            if class_type == 0:  # Head class - apply full smoothing
                smoothing_factor = current_smoothing
            elif class_type == 1:  # Medium class - apply reduced smoothing
                smoothing_factor = current_smoothing * 0.5
            else:  # Tail class - no smoothing
                smoothing_factor = 0.0
                
            # Keep most of the probability on the true class
            class_distribution = (1 - smoothing_factor) * one_hot[i]
            
            # Only distribute probability if there's any smoothing
            if smoothing_factor > 0:
                # Distribute remaining probability according to class similarity
                similarity_vector = self.similarity_matrix[label]
                class_distribution += smoothing_factor * similarity_vector
            
            smooth_labels[i] = class_distribution
            
        return smooth_labels

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

# Create standard label smoothing targets
def create_smooth_labels(labels, num_classes, smoothing=0.1):
    """
    Creates soft targets for standard label smoothing.
    Args:
        labels: True labels (batch_size)
        num_classes: Number of classes
        smoothing: Label smoothing parameter (0 means no smoothing)
    Returns:
        smooth_labels: Soft targets (batch_size, num_classes)
    """
    batch_size = labels.size(0)
    smooth_labels = torch.full(size=(batch_size, num_classes), 
                               fill_value=smoothing / (num_classes - 1),
                               device=labels.device)
    smooth_labels.scatter_(dim=1, index=labels.unsqueeze(1), value=1.0 - smoothing)
    return smooth_labels

# Warm Up with standard label smoothing
def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        feats, logits1 = net(inputs, return_features=True)
        
        # Using standard label smoothing during warmup
        smooth_labels = create_smooth_labels(labels, args.num_class, smoothing=0.1)
        loss = -torch.mean(torch.sum(F.log_softmax(logits1, dim=1) * smooth_labels, dim=1))

        logits2 = net.classify2(feats)
        loss_BCE = balanced_softmax_loss(labels, logits2, torch.tensor(cls_num_list), "mean")

        logits3 = net.classify3(feats)
        loss_BCE_3 = balanced_softmax_loss(labels, logits3, torch.tensor(cls_num_list), "mean", tau=0.5)

        L = loss + loss_BCE
        L.backward()
        optimizer.step()

        if batch_idx%50==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(args.dataset, args.imb_type, args.imb_factor, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()

def get_class_types(labels, many_shot_classes, medium_shot_classes, few_shot_classes):
    """
    Determines the class type for each instance in the batch.
    
    Args:
        labels: Class labels for the batch
        many_shot_classes: Set of head class indices
        medium_shot_classes: Set of medium class indices
        few_shot_classes: Set of tail class indices
        
    Returns:
        class_types: Tensor with 0=head, 1=medium, 2=tail for each instance
    """
    class_types = torch.zeros_like(labels)
    for i, label in enumerate(labels):
        if label.item() in many_shot_classes:
            class_types[i] = 0  # Head class
        elif label.item() in medium_shot_classes:
            class_types[i] = 1  # Medium class
        else:  # label.item() in few_shot_classes
            class_types[i] = 2  # Tail class
    return class_types

# Training function with semantic label smoothing
def train(epoch, net, optimizer, trainloader, semantic_smoother):
    net.train()
    num_iter = (len(trainloader.dataset)//args.batch_size)+1
    
    # Calculate current smoothing value - gradually increase after warmup
    if epoch <= args.warm_up:
        current_smoothing = 0.0  # No semantic smoothing during warmup
    else:
        # Linearly increase from min_smoothing to max_smoothing over 70 epochs after warmup
        progress = min(1.0, (epoch - args.warm_up) / 70.0)
        current_smoothing = args.min_smoothing + progress * (args.max_smoothing - args.min_smoothing)
    
    # Update class prototypes if using semantic smoothing
    if epoch > args.warm_up:
        semantic_smoother.update_class_prototypes(net, trainloader, epoch)
    
    for batch_idx, (inputs, labels, index) in enumerate(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        batch_size = inputs.size(0)
        
        # Feature extraction
        feats, logits = net(inputs, return_features=True)
        
        # Get outputs from different classifiers (for head, medium, tail)
        logits2 = net.classify2(feats)  # Tail expert
        logits3 = net.classify3(feats)  # Medium expert
        
        # Determine class types for each instance in the batch
        class_types = get_class_types(labels, many_shot_classes, medium_shot_classes, few_shot_classes)
        
        # Create labels based on current stage
        if epoch <= args.warm_up:
            # Use standard label smoothing during warmup
            smooth_labels = create_smooth_labels(labels, args.num_class, smoothing=0.1)
        else:
            # Use semantic label smoothing after warmup with selective application
            smooth_labels = semantic_smoother(labels, args.num_class, class_types, smoothing=current_smoothing)
        
        # Calculate loss for main classifier using smooth labels
        loss_main = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * smooth_labels, dim=1))
        
        # Apply logit adjustment for tail classifier
        spc_tail = torch.tensor(cls_num_list).cuda().type_as(logits2)
        spc_tail = spc_tail.unsqueeze(0).expand(logits2.shape[0], -1)
        adjusted_logits_tail = logits2 + args.tau_tail * (spc_tail + 1e-6).log()
        loss_tail = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_tail, dim=1) * smooth_labels, dim=1))
        
        # Apply logit adjustment for medium classifier
        spc_medium = torch.tensor(cls_num_list).cuda().type_as(logits3)
        spc_medium = spc_medium.unsqueeze(0).expand(logits3.shape[0], -1)
        adjusted_logits_medium = logits3 + args.tau_medium * (spc_medium + 1e-6).log()
        loss_medium = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_medium, dim=1) * smooth_labels, dim=1))
        
        # Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        
        # Total loss
        loss = loss_main + loss_tail + loss_medium + penalty
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%50==0:
            # Count instances by class type
            head_count = (class_types == 0).sum().item()
            medium_count = (class_types == 1).sum().item()
            tail_count = (class_types == 2).sum().item()
            
            sys.stdout.write('\r')
            sys.stdout.write('%s:%s_%g | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.2f, Tail-loss: %.2f, Medium-loss: %.2f, Smoothing: %.3f (H:%d,M:%d,T:%d)'
                    %(args.dataset, args.imb_type, args.imb_factor, epoch, args.num_epochs, batch_idx+1, num_iter, 
                      loss_main.item(), loss_tail.item(), loss_medium.item(), current_smoothing,
                      head_count, medium_count, tail_count))
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

# Initialize semantic label smoothing
semantic_smoother = SemanticLabelSmoothing(
    num_classes=args.num_class,
    max_smoothing=args.max_smoothing,
    min_smoothing=args.min_smoothing,
    similarity_temp=args.similarity_temp,
    update_freq=args.update_freq
)

# Identify many, medium, and few shot classes based on class distribution
sorted_class_counts = sorted(enumerate(cls_num_list), key=lambda x: x[1], reverse=True)
total_classes = len(cls_num_list)
top_30_percent = int(total_classes * 0.3)
bottom_30_percent = int(total_classes * 0.3)

sorted_classes = [class_idx for class_idx, _ in sorted_class_counts]
many_shot_classes = set(sorted_classes[:top_30_percent])
few_shot_classes = set(sorted_classes[-bottom_30_percent:])
medium_shot_classes = set(sorted_classes[top_30_percent: -bottom_30_percent])

print(f"Many shot classes: {many_shot_classes}")
print(f"Medium shot classes: {medium_shot_classes}")
print(f"Few shot classes: {few_shot_classes}")

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
    resume_path = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.tau_tail}_{args.tau_medium}_ep{args.resume}.pth'
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
        print('Train Net with Semantic Label Smoothing')
        train(epoch, net, optimizer, train_loader, semantic_smoother)
    
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
    print(f"Class Accuracies: {acc_dict['class_accuracy']}")
    print(f"Many Shot Accuracy: {acc_dict['many_shot_accuracy']:6.2f}%")
    print(f"Medium Shot Accuracy: {acc_dict['medium_shot_accuracy']:6.2f}%")
    print(f"Few Shot Accuracy: {acc_dict['few_shot_accuracy']:6.2f}%")

    # Save checkpoint periodically
    if epoch in [warm_up, args.num_epochs] or epoch % 50 == 0:
        save_name = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.tau_tail}_{args.tau_medium}_ep{epoch}.pth'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        print(f'| Saving model to {save_name}')
        ckpt = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(ckpt, save_name)

# Save final accuracy results
df = pd.DataFrame(acc_list)
save_name = f'./saved/{file_name}/{args.dataset}_{args.imb_factor}_{args.tau_tail}_{args.tau_medium}/Acc.csv'
os.makedirs(os.path.dirname(save_name), exist_ok=True)
df.to_csv(save_name, index=True)

# Save the best model
if best_model is not None:
    best_model_path = f'./checkpoint/{file_name}/{args.dataset}_{args.imb_factor}_{args.tau_tail}_{args.tau_medium}_best_model.pth'
    print(f"Saving the best model at epoch {best_epoch} with accuracy {best_acc:.2f}%")
    torch.save(best_model, best_model_path)