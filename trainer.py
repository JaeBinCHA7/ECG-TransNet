import torch
from utils import Bar
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np


def trainer(model,
            train_loader,
            loss_calculator,
            optimizer,
            writer,
            EPOCH,
            DEVICE
            ):
    """
    One training epoch.
    Keeps original behavior: inputs permute, model returns (outputs, feature, proxy),
    loss_calculator signature unchanged, and writer logging key names preserved.
    """
    model.train()
    train_loss = 0.0
    batch_num = 0

    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # To device; model expects [B, C, T], hence permute from [B, T, C]
        inputs = inputs.float().to(DEVICE).permute(0, 2, 1)
        targets = targets.float().to(DEVICE)

        outputs, feature, proxy = model(inputs)
        outputs = torch.squeeze(outputs)

        loss, _ = loss_calculator(feature, outputs, targets, proxy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= max(batch_num, 1)

    # Log average training loss
    writer.log_train_loss('total', train_loss, EPOCH)

    return train_loss


def validator(model,
              valid_loader,
              loss_calculator,
              writer,
              EPOCH,
              DEVICE
              ):
    """
    Validation epoch.
    Returns (avg_loss, AUPRC, AUROC) computed over the full validation set.
    """
    model.eval()
    valid_loss = 0.0
    batch_num = 0

    # Collect all targets/predictions to compute epoch-level metrics
    t_all = []
    o_all = []

    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # To device; model expects [B, C, T]
            inputs = inputs.float().to(DEVICE).permute(0, 2, 1)
            targets = targets.float().to(DEVICE)

            outputs, feature, proxy = model(inputs)
            outputs = torch.squeeze(outputs)

            # Sigmoid for probabilities
            p = torch.sigmoid(outputs)

            loss, _ = loss_calculator(feature, outputs, targets, proxy)
            valid_loss += loss.item()

            # Move to CPU numpy for sklearn
            t_all.append(targets.detach().cpu().numpy())
            o_all.append(p.detach().cpu().numpy())

    valid_loss /= max(batch_num, 1)

    # Concatenate along batch dimension
    t_all = np.concatenate(t_all, axis=0) if len(t_all) > 0 else np.zeros((0,))
    o_all = np.concatenate(o_all, axis=0) if len(o_all) > 0 else np.zeros((0,))

    # Epoch-level metrics (multi-label handling relies on sklearn defaults, same as original)
    valid_auprc = average_precision_score(y_true=t_all, y_score=o_all)
    valid_auroc = roc_auc_score(y_true=t_all, y_score=o_all)

    # Log validation stats
    writer.log_valid_loss('total', valid_loss, EPOCH)
    writer.log_score('AUPRC', valid_auprc, EPOCH)
    writer.log_score('AUROC', valid_auroc, EPOCH)

    return valid_loss, valid_auprc, valid_auroc
