from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import AminoAcidDataset
from ModelTrainer.trainer_without_overfitting import RegressorTrainerWithoutOverfit
from model.SVAE_stronger_decoder import RegressorWithStrongerDecoder

def find_pipline(
    initial_threshold=0,
    final_threshold=0,
    file_name="./data/high.csv",
    initial_index="InitialCount",
    final_index="FinalCount",
    seqs_index="AASeq",
    fitness_index="LogEnrichment",
    input_dim=17 * 20,
    latent_dim=64,
    batch_size=32,
    lr=1e-3,
    max_epoch=200,
    patience=10,
    train_valid_ratio=(0.7, 0.2),
):
    name = f"Initial_count_{initial_threshold}_final_threshold_{final_threshold}_"
    # read the data.
    dataframe = pd.read_csv(filepath_or_buffer=file_name, index_col=False)
    # filter
    dataframe = dataframe[dataframe[initial_index] > initial_threshold]
    dataframe = dataframe[dataframe[final_index] > final_threshold]

    # log
    print(f"---Running Pipline: {name}, Data Length:{dataframe.shape[0]}---")

    # df -> list
    seqs_data = dataframe[seqs_index].to_list()
    fitness_data = dataframe[fitness_index].to_list()
    # dataset.
    aa_dataset = AminoAcidDataset(seqs=seqs_data, enrichment=fitness_data)
    # split
    train_size, valid_size = [int(len(aa_dataset) * r) for r in train_valid_ratio]
    test_size = len(aa_dataset)-train_size-valid_size
    sets = random_split(dataset=aa_dataset, lengths=(train_size, valid_size, test_size))
    train_dataloader, valid_dataloader, test_dataloader = [
        DataLoader(dataset=s, batch_size=batch_size, shuffle=True)
        for s in sets
    ]

    # model and trainer.
    model = RegressorWithStrongerDecoder(input_dim=input_dim, latent_dim=latent_dim)
    aa_trainer = RegressorTrainerWithoutOverfit(model=model, lr=lr)

    # train
    train_loss_list, valid_loss_list, test_res = aa_trainer.train(
        train_dataloader, valid_dataloader, test_dataloader,
        max_epoch=max_epoch, patience=patience
    )

    # Visualize
    fig_loss = plt.Figure(figsize=(10, 6))
    res_format = ["loss", "reconstruction loss", "regression loss", "kl divergence"]
    for i, key in enumerate(res_format):
        train_loss_type = [loss_dict[key] for loss_dict in train_loss_list]
        valid_loss_type = [loss_dict[key] for loss_dict in valid_loss_list]
        ax = plt.subplot(2, 2, i+1)
        ax.plot(range(len(train_loss_type)), train_loss_type)
        ax.plot(range(len(valid_loss_type)), valid_loss_type)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(key)

    plt.tight_layout()
    plt.savefig(f"./Result_find/loss/{name}loss.png")

    # Scatter Plot
    y_true = test_res["y_true"]
    y_pred = test_res["y_predict"]
    fig_scatter = plt.Figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.scatter(y_pred, y_true, c=y_pred, cmap="viridis", s=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predict")

    # reference line
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    lower_bound = min(x_min, y_min)
    upper_bound = max(x_max, y_max)
    scatters = np.linspace(start=lower_bound, stop=upper_bound, num=20)
    ax.plot(scatters, scatters, linestyle="dashed", linewidth=1, color="grey")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(f"./Result_find/scatter/{name}scatter.png")


if __name__ == "__main__":
    # 5*5 grid search
    for i in range(6):
        for j in range(6):
            find_pipline(initial_threshold=i, final_threshold=j)
