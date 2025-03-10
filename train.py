import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from metrics.beta import getBeta
from metrics.ELBO import ELBO
from utils.logMeanExp import logMeanExp
from models.B3C3FC import B3C3FC
from data.CustomDataset import CustomDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = r"E:\DATA_Lung\Luna16.v1i.coco\train"
JSON_PATH = r"E:\DATA_Lung\Luna16.v1i.coco\train\_annotations.coco.json"


def getDataset(root, annotation_file, transform):
    dataset = CustomDataset(root=root, annotation_file=annotation_file, transform=transform)
    train_ids, val_test_ids = train_test_split(dataset.ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_ids)
    val_dataset = torch.utils.data.Subset(dataset, val_ids)
    test_dataset = torch.utils.data.Subset(dataset, test_ids)

    return train_dataset, val_dataset, dataset[0][0].shape[0], len(dataset)


def getDataloader(trainset, valset, valid_size, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, None
def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())

def train(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []

    for i, (inputs, labels) in enumerate(trainloader, 1):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze()
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = logMeanExp(outputs, dim=2)

        beta = getBeta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()

    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)

def validation(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = logMeanExp(outputs, dim=2)

        beta = getBeta(i - 1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(acc(log_outputs, labels))

    return valid_loss / len(validloader), np.mean(accs)

def run(dataset):
    activation_type = 'softplus'
    priors = {
        'prior_mu': 0,
        'prior_sigma': 0.1,
        'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
        'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
    }
    train_ens = 1
    valid_ens = 1
    n_epochs = 2
    lr_start = 0.001
    num_workers = 4
    valid_size = 0.2
    batch_size = 32
    beta_type = 0.1

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])

    trainset, valset, inputs, outputs = getDataset(DATA_PATH, JSON_PATH, transform)
    train_loader, valid_loader, _ = getDataloader(trainset, valset, valid_size, batch_size, num_workers)

    net = B3C3FC(inputs, outputs, priors, activation_type)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_B3C3FC_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf

    for epoch in range(n_epochs):
        train_loss, train_acc, train_kl = train(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                                      beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validation(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type,
                                               epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss


if __name__ == "__main__":
    dataset = "Luna16"  # Tên dataset (hoặc thay bằng tên bạn muốn)
    run(dataset)
