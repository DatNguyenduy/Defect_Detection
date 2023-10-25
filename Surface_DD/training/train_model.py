import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torchmetrics import JaccardIndex
import os


def evaluate_model(model,
                   dataloader,
                   criterion,
                   nb_classes,
                   device = torch.device("cpu"),
                   is_bar_progress = False):
    jaccard = JaccardIndex(task = "multiclass",num_classes = nb_classes).to(device)

    if is_bar_progress:
        update_progress_bar = tqdm(total = len(dataloader),desc = f"Evaluating: IoU Scoring: {np.nan:7.5f}",position=0)
    model.to(device)
    model.eval()
    count_batch = 0
    mean_iou_score = 0
    with torch.no_grad():
        for _,(X,y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            iou_score = jaccard(y_pred,y)
            mean_iou_score = (mean_iou_score*count_batch+iou_score)/(count_batch+1)
            count_batch += 1
            if is_bar_progress:
                update_progress_bar.set_description(f"Evaluating: IoU Scoring: {mean_iou_score:7.5f}", refresh = True)
                update_progress_bar.update()
        if is_bar_progress:
            update_progress_bar.close()
    model.train()
    return mean_iou_score
def training_model(results_path,
                   model_seg,
                   trainloader,
                   valloader,
                   nb_epochs,
                   nb_classes,
                   criterion_seg,
                   optimizer_seg,
                   print_status_at: int,
                   validate_at: int,
                   device = torch.device("cpu")):
    saved_model_file = os.path.join(results_path,f"best_model.pt")
    os.makedirs(results_path,exist_ok=True)
    progress_bar = tqdm(total=nb_epochs,desc=f"Training: Loss: {np.nan:7.5f}", position=0)
    write = SummaryWriter(log_dir = os.path.join(results_path,f"tensorboard"))
    best_validation_loss = np.inf
    jaccard = JaccardIndex(task = 'multiclass', num_classes = nb_classes).to(device=device)
    for epoch in range(nb_epochs):
        for batch,(X,y) in trainloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model_seg(X)
            seg_loss = criterion_seg(y_pred,y)
            # IoU score:
            iou_score = jaccard(y_pred,y)
            seg_loss.backward()
            optimizer_seg.step()
        if (epoch + 1) % print_status_at == 0: