import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

def get_average_feature_gradients(model, task_name, train_loader, criterion, device, step = 1, if_supervised = False):
    loss = 0
    count = 0
    for i, batch in enumerate(train_loader):
        if i >= step:
            break

        if if_supervised:
            data, target, _ = batch
            data, target = data.to(device), target.to(device)
            output = model(task_name, data)
            loss = criterion(output, target)
        else:
            ((data_1, data_2), target, index) = batch
            data_1, data_2 = data_1.to(device), data_2.to(device)
            h_i, h_j, z_i, z_j = model(task_name, data_1, data_2)
            loss = criterion(z_i, z_j)
        loss += loss
        count += 1
    loss = loss/count
    if hasattr(model, "encoder"):
        feature_gradients = grad(loss, model.encoder.parameters(), retain_graph=False, create_graph=False,
                                allow_unused=True)
    else:
        feature_gradients = grad(loss, model.feature_extractor.parameters(), retain_graph=False, create_graph=False,
                                allow_unused=True)
    feature_gradients = torch.cat([gradient.view(-1) for gradient in feature_gradients]) # flatten gradients
    return feature_gradients

def get_task_weights_gradients_multi(model, source_loaders, criterion, device, step=1, if_supervised=False):
    # target_gradients = get_average_feature_gradients(model, target_task, target_loader, criterion, device, step)

    source_gradients = {}
    for task, task_train_loader in source_loaders.items():
        task_gradients = get_average_feature_gradients(model, task, task_train_loader, criterion, device, step, if_supervised=if_supervised)
        source_gradients[task] = task_gradients
    
    # average source gradients to a target gradient:
    target_gradients = torch.zeros_like(task_gradients)
    count = 0
    for task, task_gradient in source_gradients.items():
        target_gradients = target_gradients + task_gradient
        count += 1
    target_gradients = target_gradients/count

    num_tasks = len(source_loaders.keys())
    task_weights_gradients = torch.zeros((num_tasks, ), device=device, dtype=torch.float)
    for i, task in enumerate(source_loaders.keys()):
        task_weights_gradients[i] = -F.cosine_similarity(target_gradients, source_gradients[task], dim=0)
    return task_weights_gradients