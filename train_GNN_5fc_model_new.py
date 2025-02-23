import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from GNN_model_new import*
from torch.utils.data import Subset, SubsetRandomSampler
from imblearn.over_sampling import SMOTE
import optuna
import argparse

## intiate randome seed
set_random_seed(seed=42)

# Entry point for the script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna Search")
    parser.add_argument('--column', type=str, required=True, help='Column name to select from NCI_CBTN_class')
    parser.add_argument('--outdir', type=str, required=False, default=None, help='Directory to save the outputs')
    #parser.add_argument('--SMOTE', type=str, choices=['TRUE', 'FALSE'], required=False, default='FALSE', help='SMOTE for imbalacnce classe')
    args = parser.parse_args()

    # Convert the string to a boolean
    #use_smote = args.SMOTE == 'TRUE'
    
    # Intiate environment
    world_size = setup_slurm_environment()
    #world_size = torch.cuda.device_count()
    '''world_size,rank,local_rank = setup_slurm_environment()
    setup_distributed_environment()
    world_size, rank, local_rank =setup_distributed_environment()'''
    print("WORLD_SIZE:",world_size)
    #print("RANK:", rank)
    #print("LOCAL_RANK:", local_rank)
    # Generate the output directory automatically if not provided
    if args.outdir:
        outdir=args.outdir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir = f"./MLP_Bagging_model_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    print(f"loading data!")

    ## load NCI CBTN Group 3 and 4 features and classes
    slide_features=np.load("../features/NCI_CBTN_GP34_features.npy", allow_pickle=True)
    slide_classes=pd.read_csv("../features/NCI_CBTN_GP34_Malignant&Myeloid_Cellstate_classes.csv",index_col=0)

    

    # Seperate sample names and image tile level features
    sample_names, features = extract_sample_names_and_features(slide_features)

    # Scale the features
    scaled_features = scale_tile_features(features)

    # Encode the target classes
    slide_classes=slide_classes[args.column]
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(slide_classes)

    # creating graph data from tile level featutes
    graph_dataset = TileGraphDataset(scaled_features, y_labels, k=5)

    print(len(graph_dataset))
    print(graph_dataset[0])

    #train_dataset, val_dataset = split_graph_data(graph_dataset, train_percent=80, val_percent=20)

    print(f"Data spliting is done...!")

    # Hyperparameter tuning
    #study = optuna.create_study(direction="maximize")
    #study.optimize(lambda trial: objective(trial, scaled_features, y_labels, k=5), n_trials=50)
    
    torch.multiprocessing.spawn(
        distributed_hyperparameter_search,
        args=(world_size, graph_dataset,  50),
        nprocs=world_size,
        join=True,
    )
    #distributed_hyperparameter_search(rank, world_size,local_rank, graph_dataset,  50)

    # Best hyperparameters
    print("Best hyperparameters:", study.best_params)

    # Train final model
    best_params = study.best_params

    folds = stratified_kfold_split(graph_dataset, labels, num_folds=5)

    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Create a directory for this fold
        fold_dir = os.path.join(outdir,f"result_{fold}")
        os.makedirs(fold_dir, exist_ok=True)  # Create folder if it doesn't exist

        train_dataset = Subset(graph_dataset, train_idx)
        val_dataset = Subset(graph_dataset, val_idx)

       
        
        input_dim = len(graph_dataset[0][1])
        hidden_dim = best_params["hidden_dim"]
        num_classes = len(np.unique(y_labels))
        num_epochs = best_params["num_epochs"]
        batch_size = best_params["batch_size"]
        dropout = best_params["dropout"]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        print("checking train loder  ...!")
        check_batches(train_loader)
        print("Checking validation loader....!")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        check_batches(val_loader)

        # define device GPU or CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,dropout=dropout)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        # **Calculate class weights based on the training data**
        #y_train = torch.tensor(y_train, dtype=torch.long)
        class_weights = sk_class_weights(y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

        # Train and evaluate
        accuracy, val_preds, val_labels, val_scores = train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=best_params["num_epochs"])
        fold_accuracies.append(accuracy)

        # Save model for this fold
        model_path = os.path.join(fold_dir, "GNN_ViT_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model for Fold {fold + 1} saved at {model_path}")

        # Save results for this fold in a text file
        save_results_to_text(fold_dir, sample_names, val_idx, val_labels, val_preds, val_scores, label_encoder)
  
        print(f"Fold {fold + 1} Accuracy: {accuracy}")

    print(f"Mean Accuracy across folds: {sum(fold_accuracies) / len(fold_accuracies)}")

    # Collect data
    true_labels, predicted_labels, predicted_scores = collect_and_combine_results(args.outdir)

    # Annalyze data
    analyze_results(true_labels, predicted_labels, predicted_scores, label_encoder,outdir=args.outdir)
   
    print("All GNN and ViT hybride modle 5 fold cross validation process is Done...!")
