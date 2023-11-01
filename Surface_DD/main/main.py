import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append("..")
import numpy as np

import torch
from torch.utils.data import DataLoader,Subset
import torchvision.transforms as transforms

import torch.nn as nn

from dataset.defect_dataset import DefectDataset
from model.models import UNet

from training.train_model import training_model

def main(results_path,
        unet_config:dict,
        data_path,
        image_size = (256,128),
        nb_classes = 1,
        batch_size = 32,
        num_workers:int = 1,
        learningrate: int = 1e-3, 
        weight_decay: float = 1e-5,
        nb_epochs: int = 500
    ):
    np.random.seed(0)
    torch.manual_seed(0)
    tr_forms = transforms.ToTensor()

    # Load or download source dataset
    source_dl = DefectDataset(data_dir=data_path,img_size=image_size)
    # source_dl = Subset(source_dl,indices=np.arange(int(len(source_dl)*0.1)))
    #source_dl = SourceData(**data_config,train_transformer(**data_config))
    
    # Split source into training, validation and test set
    trainingset = Subset(source_dl,indices=np.arange(int(len(source_dl) * (4 / 5))))
    validationset = Subset(source_dl, indices=np.arange(int(len(source_dl) * (4 / 5)), len(source_dl)))


    print(f"[INFO] found {len(trainingset)} examples in the SOURCE training set...")
    print(f"[INFO] found {len(validationset)} examples in the source validationset set...")
    
    # Create datasets and dataloaders 
    trainloader = DataLoader(trainingset, batch_size, shuffle=False, num_workers=num_workers)
    valloader = DataLoader(validationset, batch_size, shuffle=False, num_workers=num_workers)
    
    
    # Create Network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device is {device}")
    model_Seg = UNet(**unet_config).to(device)

    # Get loss function
    criterion_seg = nn.CrossEntropyLoss()

    # Get adam optimizer
    optimizer_seg = torch.optim.Adam(
            model_Seg.parameters(),
            lr=learningrate,
            weight_decay=weight_decay, betas=(0.5, 0.99)
        )

    print_stats_at = int(nb_epochs/10)  # print status to tensorboard every x updates
    validate_at = int(nb_epochs/10)  # evaluate model on validation set and check for new best model every x updates
    # Train model ans save the best model and return saved model file path
    saved_model_file_path = training_model(results_path=results_path,
                                           model_seg=model_Seg,
                                           
                                           trainloader=trainloader,
                                           valloader=valloader,
                                           
                                           nb_epochs=nb_epochs,
                                           nb_classes=nb_classes,
                                           criterion_seg=criterion_seg,
                
                                           optimizer_seg=optimizer_seg,
                                    
                                           print_status_at=print_stats_at,
                                           validate_at=validate_at,
                                       
                                           device=device,
                                           )



    # # Load best model and compute score on test set
    # load_best_model(results_path=results_path,
    #                 saved_model_file_path=saved_model_file_path,
    #                 train_loader=source_trainloader,
    #                 val_loader=source_valloader,
    #                 test_loader=None,
    #                 nb_classes=nb_classes,
    #                 criterion=criterion_seg,
    #                 lambda_=lambda_,
    #                 device=device)


if __name__ == "__main__":
    import argparse
    import json
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)