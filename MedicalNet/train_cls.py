'''
Training code for PCR classification with post DCE volumes
'''

from setting_cls import parse_opts
from datasets.pcr_dce_cls import PcrDceClsDataset
from model_cls import generate_model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
import os


def train_one_epoch(data_loader, model, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    correct = 0
    total = 0

    for volumes, labels in data_loader:
        volumes = volumes.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(volumes)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate(data_loader, model, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for volumes, labels in data_loader:
            volumes = volumes.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            logits = model(volumes)
            loss = loss_fn(logits, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def main():
    sets = parse_opts()
    torch.manual_seed(sets.manual_seed)

    if sets.no_cuda:
        device = torch.device('cpu')
        sets.pin_memory = False
    else:
        device = torch.device('cuda')
        sets.pin_memory = True

    model, parameters = generate_model(sets)

    if sets.ci_test:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100},
        ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    if sets.resume_path and os.path.isfile(sets.resume_path):
        checkpoint = torch.load(sets.resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    sets.phase = 'train'
    train_dataset = PcrDceClsDataset(sets.img_list, sets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=sets.batch_size,
        shuffle=True,
        num_workers=sets.num_workers,
        pin_memory=sets.pin_memory)

    val_loader = None
    if os.path.isfile(sets.val_list):
        sets.phase = 'val'
        val_dataset = PcrDceClsDataset(sets.val_list, sets)
        val_loader = DataLoader(
            val_dataset,
            batch_size=sets.batch_size,
            shuffle=False,
            num_workers=sets.num_workers,
            pin_memory=sets.pin_memory)

    for epoch in range(sets.n_epochs):
        scheduler.step()
        train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, device)
        msg = "Epoch {} | train loss {:.4f} acc {:.4f}".format(epoch, train_loss, train_acc)
        if val_loader is not None:
            val_loss, val_acc = evaluate(val_loader, model, device)
            msg += " | val loss {:.4f} acc {:.4f}".format(val_loss, val_acc)
        print(msg)

        if not sets.ci_test and epoch % sets.save_intervals == 0 and epoch != 0:
            model_save_path = '{}_epoch_{}.pth.tar'.format(sets.save_folder, epoch)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)


if __name__ == '__main__':
    main()
