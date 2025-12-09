# VTS (DSH with ViT Backbone - ICME 2022)
# paper [Vision Transformer Hashing for Image Retrieval, ICME 2022](https://arxiv.org/pdf/2109.12564.pdf)
# DSH basecode considered from https://github.com/swuxyj/DeepHash-pytorch

from utils.tools import *
from network import *
import os
import torch
import torch.optim as optim
import time
import numpy as np
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
from vim_mamba_model import VisionMambaHashing, VIM_CONFIGS # ADDED: ViM imports
import random
torch.multiprocessing.set_sharing_strategy('file_system')

def get_config():
    config = {
        "dataset": "cifar10",
        #"dataset": "cifar10-2",
        #"dataset": "coco",
        #"dataset": "nuswide_21",
        # "dataset": "imagenet",
        
        "net": VisionMambaHashing, "net_print": "ViM-T_16", "model_type": "ViM-T_16", "pretrained_dir": "pretrainedVIM/ViM-T_16.npz",
        # "net": VisionMambaHashing, "net_print": "ViM-S_16", "model_type": "ViM-S_16", "pretrained_dir": "pretrainedVIM/ViM-S_16.npz",
        # "net": VisionMambaHashing, "net_print": "ViM-B_16", "model_type": "ViM-B_16", "pretrained_dir": "pretrainedVIM/ViM-B_16.npz",
 
        "bit_list": [32,16],
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "device": torch.device("cuda"), "save_path": "Checkpoints_Results",
        "epoch": 150, "test_map": 30, "batch_size": 64, "resize_size": 256, "crop_size": 224,
        "info": "DSH", "alpha": 0.1,
    }
    config = config_dataset(config)
    return config

def train_val(config, bit):
    start_epoch = 1
    Best_mAP = 0
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    
    num_classes = config["n_class"]
    hash_bit = bit
    
    if "ViT" in config["net_print"]:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](vit_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    # ADDED: Logic for VisionMambaHashing
    elif "ViM" in config["net_print"]:
        vim_config = VIM_CONFIGS[config["model_type"]] 
        net = config["net"](vim_config, config["crop_size"], zero_head=True, num_classes=num_classes, hash_bit=hash_bit).to(device)
    # END ADDED
    else:
        net = config["net"](bit).to(device)
    
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])
    best_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-BestModel.pt")
    trained_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + "-IntermediateModel.pt")
    results_path = os.path.join(config["save_path"], config["dataset"] + "_" + config["info"] + "_" + config["net_print"] + "_Bit" + str(bit) + ".txt")
    f = open(results_path, 'a')
    
    if os.path.exists(trained_path):
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(trained_path)
        net.load_state_dict(checkpoint['net'])
        Best_mAP = checkpoint['Best_mAP']
        start_epoch = checkpoint['epoch'] + 1
    else:
        # MODIFIED: Update logic for pretrained loading to include "ViM"
        if "ViT" in config["net_print"] or "ViM" in config["net_print"]:
            print('==> Loading from pretrained model..')
            net.load_from(np.load(config["pretrained_dir"]))
    
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = DSHLoss(config, bit)

    for epoch in range(start_epoch, config["epoch"]+1):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["net_print"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u = net(image)
            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        f.write('Train | Epoch: %d | Loss: %.3f\n' % (epoch, train_loss))

        if (epoch) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            
            if mAP > Best_mAP:
                Best_mAP = mAP
                P, R = pr_curve(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy())
                print(f'Precision Recall Curve data:\n"DSH":[{P},{R}],')
                f.write('PR | Epoch %d | ' % (epoch))
                for PR in range(len(P)):
                    f.write('%.5f %.5f ' % (P[PR], R[PR]))
                f.write('\n')
            
                print("Saving in ", config["save_path"])
                state = {
                    'net': net.state_dict(),
                    'Best_mAP': Best_mAP,
                    'epoch': epoch,
                }
                torch.save(state, best_path)
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch, bit, config["dataset"], mAP, Best_mAP))
            f.write('Test | Epoch %d | MAP: %.3f | Best MAP: %.3f\n'
                % (epoch, mAP, Best_mAP))
            print(config)
            
            state = {
                'net': net.state_dict(),
                'Best_mAP': Best_mAP,
                'epoch': epoch,
            }
            torch.save(state, trained_path)
    f.close()


class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        # The penalty term in DSH is often |u^2 - 1|, which is similar to the quantization loss (1 - |u|) used in the original HashNet paper, 
        # but the DSH basecode you provided uses a loss term which can be simplified as just the magnitude of the difference from 1. 
        # The basecode provided in the prompt uses: config["alpha"] * (1 - u.sign()).abs().mean(). This is unusual for DSH; a common DSH loss 
        # includes a quantization term like (u.abs() - 1)^2 or similar. 
        # The HashNet code used config["alpha"] * (1 - u.sign()).abs().mean(). Since you want to match the *HashNet* structure, 
        # I'll keep the DSH basecode's loss: config["alpha"] * (1 - u.sign()).abs().mean()
        loss2 = config["alpha"] * (1 - u.sign()).abs().mean() 
        # Note: If the intention was to use a *standard* DSH quantization loss, it would typically be a simplified version of:
        # loss2 = config["alpha"] * (u.abs() - 1).pow(2).mean() 
        # But I'm adhering to the basecode provided in the DSH loss.

        return loss1 + loss2


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)