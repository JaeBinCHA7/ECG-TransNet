import os
import argparse
import time
import random
from datetime import datetime
import numpy as np
import torch
from colorama import Fore, Style, init
import options
import utils
from dataloader import ECGDataModule
from trainer import trainer, validator
from model import ECGTransNet
from loss import HeadwiseProxyBCELoss

# Initialize colorama for colored console output
init(autoreset=True)


def build_log_dir(opt) -> str:
    """
    Build a descriptive log directory path including run-time hyper-params.
    Keeps the original naming scheme and fields.
    """
    dir_name = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(
        dir_name,
        'log',
        f"{opt.log_name}_{timestamp}_{opt.arch}_bs{opt.batch_size}_lr{opt.lr_initial}_"
        f"E{opt.d_model}_H{opt.head}_Proxy_w{opt.proxy_weight}_a{opt.proxy_a}_d{opt.proxy_d}"
    )
    return log_dir


def save_options(opt, model_dir: str) -> None:
    """Dump parsed options to options.txt for reproducibility."""
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, 'options.txt')
    with open(file_path, 'w') as f:
        for key, value in vars(opt).items():
            f.write(f'{key}: {value}\n')


def set_seeds() -> None:
    """Set random seeds and cuDNN flags (same values as original)."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Keep original cuDNN settings (speed-oriented, non-deterministic)
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def maybe_resume(opt, model, optimizer, pretrained_model_path: str):
    """
    If --pretrained is set, resume model/optimizer states and epoch index.
    Returns the starting epoch index (1-based), matching original semantics.
    """
    epoch_start_idx = 1
    if opt.pretrained:
        print(Fore.CYAN + 'Load the pretrained model...')
        chkpt = torch.load(pretrained_model_path, weights_only=False)
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        epoch_start_idx = chkpt['epoch'] + 1
        print(Fore.CYAN + f'Resuming Start Epoch: {epoch_start_idx}')
        utils.optimizer_to(optimizer, torch.device(opt.device))
        # Reset LR to current run's initial LR (kept from original code)
        optimizer.param_groups[0]['lr'] = opt.lr_initial
    return epoch_start_idx


def main():
    ###########################################################################
    # Parser init
    ###########################################################################
    # Parse command-line arguments and configurations for the training experiment
    opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()
    print(opt)

    ###########################################################################
    # Log / model directories
    ###########################################################################
    dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
    print(dir_name)

    log_dir = build_log_dir(opt)
    utils.mkdir(log_dir)
    tboard_dir = os.path.join(log_dir, 'logs')
    model_dir = os.path.join(log_dir, 'models')
    utils.mkdir(model_dir)
    utils.mkdir(tboard_dir)

    # Save options and back up model file
    save_options(opt, model_dir)

    ###########################################################################
    # Model / loss / optimizer / scheduler
    ###########################################################################
    DEVICE = torch.device(opt.device)
    set_seeds()

    model = ECGTransNet(opt=opt)
    loss_calculator = HeadwiseProxyBCELoss(opt).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_initial, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 15, 20, 25], gamma=opt.gamma
    )

    total_params = utils.cal_total_params(model)
    print(Fore.YELLOW + 'total params (gen)  : %d (%.2f M, %.2f MBytes)\n' %
          (total_params, total_params / 1_000_000.0, total_params * 4.0 / 1_000_000.0))

    # resume if requested
    epoch_start_idx = maybe_resume(opt, model, optimizer, opt.pretrained_model)

    ###########################################################################
    # Data
    ###########################################################################
    data_module = ECGDataModule(opt)
    train_loader = data_module.get_train_loader()
    valid_loader = data_module.get_valid_loader()

    print(Fore.GREEN + "Sizeof training set: ", train_loader.__len__(),
          ", sizeof validation set: ", valid_loader.__len__())

    model = model.to(DEVICE)

    ###########################################################################
    # Training loop
    ###########################################################################
    writer = utils.Writer(tboard_dir)
    train_log_fp = open(os.path.join(model_dir, 'train_log.txt'), 'a')

    # Keep individual bests for reference; saving criterion is best (AUPRC + AUROC)
    max_auroc = 0.0
    max_auprc = 0.0
    best_sum = float('-inf')

    print(Fore.MAGENTA + 'Train start...')

    for epoch in range(epoch_start_idx, opt.nepoch + 1):
        st_time = time.time()

        # Train for one epoch
        train_loss = trainer(model, train_loader, loss_calculator, optimizer, writer, epoch, DEVICE)

        # Validate
        valid_loss, auprc, auroc = validator(
            model, valid_loader, loss_calculator, writer, epoch, DEVICE)

        # Step LR scheduler (kept after validation, as in original)
        scheduler.step()

        # Print current LRs (kept exact semantics; note: printed label uses epoch+1 originally)
        lr_list = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {epoch + 1}: LRs = {lr_list}")

        # Save checkpoint if sum metric improves
        combo = auprc + auroc
        if combo >= best_sum:
            best_sum = combo
            max_auprc = max(max_auprc, auprc)
            max_auroc = max(max_auroc, auroc)
            save_path = os.path.join(model_dir, 'best_model.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'auprc': auprc,
                'auroc': auroc,
                'combo': combo,
                'best_sum': best_sum
            }, save_path)
            print(Fore.YELLOW + f'>> New best (AUPRC+AUROC={combo:.6f}) model saved at epoch {epoch} '
                                f'[AUPRC={auprc:.6f}, AUROC={auroc:.6f}]')

        # Logging
        elapsed = time.time() - st_time
        msg = (f'EPOCH[{epoch}] T {train_loss:.6f} | V {valid_loss:.6f} | '
               f'AUPRC {auprc:.6f} | AUROC {auroc:.6f} | SUM {combo:.6f}  takes {elapsed:.3f} seconds')
        print(Fore.BLUE + msg)
        train_log_fp.write(msg + '\n')

    print(Fore.RED + 'Training has been finished.')
    train_log_fp.close()


if __name__ == '__main__':
    main()
