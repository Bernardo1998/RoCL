import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torch
import torch.nn as nn

from resnet import resnet18
from resnetROCL import ResNet18

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, normalize=None): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
    
    # Normalize to z-scores
    if normalize == "z-score":
        bins = 40
        #plt.hist(tensor.view(-1).numpy(),bins)
        #hist = torch.histc(tensor, bins = 40, min = tensor.min(), max = tensor.max())
        #x = range(bins)
        #plt.bar(x, hist, align='center')
        #plt.xlabel('Bins')
        q75 = torch.quantile(tensor, 0.75)
        q25 = torch.quantile(tensor, 0.25)
        tensor = (tensor - tensor.median()) / (q75-q25)
    elif normalize == "log":
        tensor = (tensor - tensor.min() + 5000) / (tensor.std())
        tensor = tensor.log()
    elif normalize == "byfilter":
        for i in range(64):
            tensor[i] = (tensor[i] - tensor[i].min()) / (tensor[i].max() - tensor[i].min())

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    print(tensor.max(), tensor.mean(), tensor.shape, grid.max(), grid.mean(), grid.shape)
    filter_checking = 4
    print(tensor[filter_checking,:,:,:].max(), tensor[filter_checking,:,:,:].mean(),tensor[filter_checking,:,:,:].shape)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    layer = 1

    #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/adv pretraining cifar 100/model_399.pth"
    #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar100 400/model_399.pth"
    #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/adv contrastive cifar10/model_399.pth"
    
    RoCL = True
    contrast = True
    num_classes=10
    remove_module = True
    robust = False
    if RoCL:
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/checkpoint/ckpt.t7testLog.txtRep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-10_b256_nGPU1_l256_0_epoch_230"
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/checkpoint/ckpt.t7testLog.txtsupervised_ResNet18_cifar-10_b256_nGPU1_l256_0"
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/adv_pretrain_400/ckpt.t7contrast_eval_Evaluate_linear_eval_ResNet18_cifar-10_42"
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/advsup_pretrain_400/ckpt.t7advSupEval2_Evaluate_linear_eval_ResNet18_cifar-100_42"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/adv pretraining cifar 10/resnetRoCL/model_249.pth"
        # CIFAR10 to CIFAR10: contrastive
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar10_to_cifar10/clean_pretrain_200/ckpt.t7testLog.txtsupervised_ResNet18_cifar-10_b256_nGPU1_l256_0_epoch_200"  # cons=F,rem=F, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar10_to_cifar10/clean_pretrain_400/ckpt.t7testLog.txtsupervised_ResNet18_cifar-10_b256_nGPU1_l256_0" # cons=F,rem=F, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar10_to_cifar10/adv_downstreamwithadv_linear_1400/ckpt.t7constrast_adveval2_Evaluate_linear_eval_ResNet18_cifar-10_42" #cons=T, rem=T, rob=F
        # CIFAR10 to CIFAR10: adv supervised, resnetRoCL, robust
        #model_path = "/media/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar10 200/resnetRoCL/model_199.pth"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar10 400/RoCL model/model_399.pth" # rob=True
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean downstream cifar10 after cifar10/resnetrocl/model_599.pth"
        # CIFAR10 to CIFAR10: adv supervised, resnetRoCL, ROCL only
        #model_path = '/media/xflin/10tb/Xiaofeng/RoCL/src/trained_models/cifar10_to_cifar10/clean_pretrain_200/ckpt.t7testLog.txtsupervised_ResNet18_cifar-10_b256_nGPU1_l256_0_epoch_200' # cons=F,rem=F, rob=F clean(old)
        #model_path = '/media/xflin/10tb/Xiaofeng/RoCL/src/trained_models/restart/cifar10_to_cifar10/clean_pretrain_200/ckpt.t7testLog.txtsupervised_ResNet18_cifar-10_b256_nGPU1_l256_0_epoch_200' # cons=F,rem=T, rob=F clean(new)
        model_path = "/media/xflin/10tb/Xiaofeng/RoCL/src/trained_models/cifar10_to_cifar10/adv_pretrain_400/advSup/ckpt.t7advSupRoCL_Evaluate_linear_eval_ResNet18_cifar-10_42" # cons=T,rem=F, rob=F adv pretrain
        # CIFAR100 To CIFAR10: adv supervised, resnetRoCL
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_200/transform updated/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0_epoch_200" # cons=F,rem=F, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_400/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0_epoch_400" # cons=T,rem=F, rob=F
        #model_path = '/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_downstream_600/ckpt.t7advSupCleanDownstream2_Evaluate_linear_eval_ResNet18_cifar-10_42' #cons=T, rem=T, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/adv_pretrain_400_fromrobust/model_240.pth" # No longer use this.
        # CIFAR100 TO CIFAR10: adv contrast, resnetRoCL
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_200/transform updated/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0_epoch_200" # cons=F,rem=F, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_400/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0_epoch_400" # cons=T,rem=F, rob=F
        #model_path = "/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/adv_pretrain_1200/ckpt.t7testLog.txtRep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-100_b256_nGPU1_l256_0" # cons=T,rem=F, rob=F
        #model_path = '/home/xflin/Xiaofeng/RoCL/src/checkpoint/ckpt.t7testLog.txtRep_attack_ep_0.0314_alpha_0.007_min_val_0.0_max_val_1.0_max_iters_7_type_linf_randomstart_Truecontrastive_ResNet18_cifar-100_b256_nGPU1_l256_0'
        # CHANGE in CIFAR100 contrastive overtime
        #model_path = '/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_200/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0'
        #model_path = '/home/xflin/Xiaofeng/RoCL/src/trained_models/cifar100_to_ciar10/clean_pretrain_200/ckpt.t7testLog.txtsupervised_ResNet18_cifar-100_b256_nGPU1_l256_0'
        model = ResNet18(num_classes,contrast)
        state_dict = torch.load(model_path)
        if not robust:
            state_dict = state_dict['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if remove_module:
                name = k[7:]
            else:
                name = k
            name = name.replace("linear", "fc")
            if contrast and ("fc" in k or "linear" in k):
                continue
            new_state_dict[name] = v
        #new_state_dict['fc.weight'] = model.fc.weight
        #new_state_dict['fc.bias'] = model.fc.bias
        model.load_state_dict(new_state_dict)
    else:
        contrast = False
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar10 200/model_199.pth"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/adv pretraining cifar 10/model_399.pth"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/adv pretraining cifar 100/model_399.pth"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar100 400/model_399.pth"
        #model_path = "/home/xflin/Xiaofeng/robust_overfitting/cifar_checkpoints/clean pretraining cifar100 200/model_199.pth"
        model = resnet18(False, True, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        
        
    fname = "resnet18_layer{}.png".format(layer)
    filter = model.conv1.weight.data.clone().cpu()
    
    visTensor(filter, ch=0, allkernels=False, normalize="byfilter")

    plt.axis('off')
    #plt.ioff()
    plt.show()
