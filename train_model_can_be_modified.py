from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data.dataset import AminoAcidDataset
from ModelTrainer.trainer_overfit import RegressorTrainer
from model.SVAE_stronger_decoder import RegressorWithStrongerDecoder, RegressorWithConvolution

# -----------TrainParameters-----------#
initial_threshold = 0
final_threshold = 0
file_name = "./data/high.csv"
initial_index = "InitialCount"
final_index = "FinalCount"
seqs_index = "AASeq"
fitness_index = "LogEnrichment"
input_dim = 17 * 20
latent_dim = 64
batch_size = 32
lr = 1e-3
max_epoch = 200
patience = 10
# -----------TrainParameters-----------#

dataframe = pd.read_csv(filepath_or_buffer=file_name, index_col=False)
# filter
dataframe = dataframe[dataframe[initial_index] > initial_threshold]
dataframe = dataframe[dataframe[final_index] > final_threshold]

# df -> list
seqs_data = dataframe[seqs_index].to_list()
fitness_data = dataframe[fitness_index].to_list()
# dataset.
aa_dataset = AminoAcidDataset(seqs=seqs_data, enrichment=fitness_data)
aa_dataloader = DataLoader(aa_dataset, batch_size=batch_size, shuffle=True)

# model and trainer.
model = RegressorWithStrongerDecoder(input_dim=input_dim, latent_dim=latent_dim)
# model = RegressorWithConvolution(input_dim=input_dim, latent_dim=latent_dim)
aa_trainer = RegressorTrainer(model=model, lr=lr)
# train.
train_loss_list, valid_res = aa_trainer.train(aa_dataloader, aa_dataloader, max_epoch=max_epoch, patience=patience)

res_format = ["loss", "reconstruction loss", "regression loss", "kl divergence"]
fig_train_loss = plt.Figure(figsize=(10, 6))
for i, key in enumerate(res_format):
    loss_type = [loss_dict[key] for loss_dict in train_loss_list]
    ax = plt.subplot(2, 2, i+1)
    ax.plot(range(len(loss_type)), loss_type)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Train_"+key)
plt.tight_layout()
plt.show()

# scatter plot
y_true = valid_res["y_true"]
y_pred = valid_res["y_predict"]
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

plt.show()
