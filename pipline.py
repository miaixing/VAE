import torch
from torch import nn
import copy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchmetrics.functional import r2_score, pearson_corrcoef
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, one_hot_seq: torch.Tensor):
        return self.main(one_hot_seq)

class Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, latent_z: torch.Tensor):
        return self.main(latent_z)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder_mean = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.encoder_logvar = copy.deepcopy(self.encoder_mean)
        self.decoder = Decoder(input_dim=latent_dim, latent_dim=input_dim)

    @staticmethod
    def re_parameter_trick(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean+eps * std

    @staticmethod
    def kl_divergence(mean, logvar):
        kl_div = -0.5 * torch.sum(1+logvar-mean.pow(2)-logvar.exp(), dim=1).mean()
        return kl_div

    def forward(self, one_hot_seq: torch.Tensor):
        mean = self.encoder_mean(one_hot_seq)
        log_var = self.encoder_logvar(one_hot_seq)
        latent_z = self.re_parameter_trick(mean, log_var)
        x_recons = self.decoder(latent_z)

        # loss
        kl_divergence = self.kl_divergence(mean, log_var)
        recons_loss = nn.functional.mse_loss(input=x_recons, target=one_hot_seq)
        loss = kl_divergence+recons_loss

        return latent_z, loss, kl_divergence, recons_loss

class EnrichmentRegressor(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(EnrichmentRegressor, self).__init__()
        self.vae = VAE(input_dim, latent_dim)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim+latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, one_hot_seq: torch.Tensor, target=None):
        z, loss, kl_loss, recons_loss = self.vae(one_hot_seq)
        fusion = torch.cat([one_hot_seq, z], dim=1)
        enrichment = self.regressor(fusion)

        if target is not None:
            reg_loss = nn.functional.mse_loss(input=enrichment.view(-1), target=target)
        else:
            reg_loss = 0

        return enrichment, loss+reg_loss, kl_loss, recons_loss, reg_loss

class AminoAcidDataset(Dataset):
    def __init__(self, seqs, enrichment):
        self.seqs = seqs
        self.enri = enrichment
        self.amino_acids = list("ARNDCQEGHILKMFPSTWYV")
        self.vocab = dict(
            zip(self.amino_acids, range(len(self.amino_acids)))
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_ids = torch.tensor([self.vocab[i] for i in self.seqs[index]])
        seq_one_hot = nn.functional.one_hot(seq_ids, num_classes=len(self.vocab)).to(dtype=torch.float32).view(-1)

        return seq_one_hot, self.enri[index]

class RegressorTrainer(object):
    def __init__(self, model: nn.Module, lr=1e-3, device="cpu"):
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device

    def run_train_epoch(self, dataloader: DataLoader):
        self.model.train()
        loss, recon_loss, regress_loss, kl_loss = 0, 0, 0, 0

        with tqdm(total=len(dataloader), desc="Train") as pbar:
            for x, y in dataloader:
                x, y = x.to(device=self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
                enrichment, total_loss, kl_div, recons_loss, reg_loss = self.model(x, y)

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                loss += total_loss.item()
                recon_loss += recons_loss.item()
                regress_loss += reg_loss.item()
                kl_loss += kl_div.item()

                pbar.update(1)
        print(
            f"\nTotal Loss: {loss:.4f}, Reconstruction Loss: {recon_loss:.4f}, "
            f"KL Loss: {kl_loss:.4f}, Regression Loss: {regress_loss:.4f}"
        )
        return_format = [
            "loss", "reconstruction loss", "regression loss", "kl divergence"
        ]
        train_result = dict(zip(
            return_format, [loss, recon_loss, regress_loss, kl_loss]
        ))
        return train_result

    def run_valid_epoch(self, dataloader: DataLoader):
        self.model.eval()
        loss, recon_loss, regress_loss, kl_loss = 0, 0, 0, 0
        enrichment_true = None
        enrichment_pred = None
        with tqdm(total=len(dataloader), desc="Train") as pbar:
            with torch.no_grad():
                for x, y in dataloader:
                    x, y = x.to(device=self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
                    enrichment, total_loss, kl_div, recons_loss, reg_loss = self.model(x, y)
                    loss += total_loss.item()
                    recon_loss += recons_loss.item()
                    regress_loss += reg_loss.item()
                    kl_loss += kl_div.item()
                    # update the progress bar.
                    pbar.update(1)

                    if enrichment_pred is None:
                        enrichment_pred = enrichment
                        enrichment_true = y
                    else:
                        enrichment_pred = torch.cat((enrichment_pred, enrichment), dim=0)
                        enrichment_true = torch.cat((enrichment_true, y), dim=0)
        # print(
        #     f"Total Loss: {total_loss:.4f}, Reconstruction Loss: {recon_loss:.4f}, "
        #     f"KL Loss: {kl_loss:.4f}, Regression Loss: {regress_loss:.4f}"
        # )
        r2 = r2_score(preds=enrichment_pred.view(-1), target=enrichment_true).item()
        pearson = pearson_corrcoef(preds=enrichment_pred.view(-1), target=enrichment_true).item()
        print(f"--R2 {r2:.3f} Pearson {pearson:.3f}---")

        return_format = [
            "loss", "reconstruction loss", "regression loss", "kl divergence", "y_true", 'y_predict', "r2", "pearson"
        ]
        valid_result = dict(zip(
            return_format,
            [loss, recon_loss, regress_loss, kl_loss, enrichment_true.tolist(), enrichment_pred.tolist(), r2, pearson]
        ))
        return valid_result

    def train(self, train_dataloader, valid_dataloader, max_epoch=100, patience=10):
        loss_list = []
        true_patience, run_epoch = 0, 0
        train_dict = None
        while true_patience <= patience and run_epoch <= max_epoch:
            print(f"--Running the {run_epoch} epoch--\n")
            train_dict = self.run_train_epoch(train_dataloader)
            # record
            loss_list.append(train_dict)

            if len(loss_list) > 2 and loss_list[-1]["regression loss"] > loss_list[-2]["regression loss"]:
                print("converged")
                true_patience += 1
            else:
                print("Not converged")
            run_epoch += 1

        valid_dict = self.run_valid_epoch(valid_dataloader)

        return loss_list, valid_dict

def pipline(
        initial_threshold: int, final_threshold: int, file_name: str = "./data/清洗过的数据-7AA.csv",
        seqs_index: str = "realAAseq", fitness_index: str = "fitness",
        initial_index: str = "plasmid_counts", final_index: str = "start_virus_counts",
        input_dim: int = 7*20, latent_dim: int = 32,
        batch_size: int = 32, lr: float = 1e-3, max_epoch: int = 32, patience: int = 10,
):
    # Define the name of the assay.
    name = f"Initial_count_{initial_threshold}_final_threshold_{final_threshold}_"
    # load the data.
    dataframe = pd.read_csv(filepath_or_buffer=file_name, index_col=False)
    # filter
    dataframe = dataframe[dataframe[initial_index] > initial_threshold]
    dataframe = dataframe[dataframe[final_index] > final_threshold]
    print(f"---Running Pipline: {name}, Data Length:{dataframe.shape[0]}---")
    # df -> list
    seqs_data = dataframe[seqs_index].to_list()
    fitness_data = dataframe[fitness_index].to_list()
    # dataset.
    aa_dataset = AminoAcidDataset(seqs=seqs_data, enrichment=fitness_data)
    aa_dataloader = DataLoader(aa_dataset, batch_size=batch_size, shuffle=True)
    # model and trainer.
    model = EnrichmentRegressor(input_dim=input_dim, latent_dim=latent_dim)
    aa_trainer = RegressorTrainer(model=model, lr=lr)
    # train.
    train_loss_list, valid_res = aa_trainer.train(aa_dataloader, aa_dataloader, max_epoch=max_epoch, patience=patience)
    # unpack the dict.
    # train_loss, train_recon_loss, train_regress_loss, train_kl_loss = [
    #     train_res[key] for key in ["loss", "reconstruction loss", "regression loss", "kl divergence"]
    # ]
    res_format = ["loss", "reconstruction loss", "regression loss", "kl divergence"]
    fig_train_loss = plt.Figure(figsize=(10, 6))
    for i, key in enumerate(res_format):
        loss_type = [loss_dict[key] for loss_dict in train_loss_list]
        # ax = fig_train_loss.add_subplot(2, 2, i+1)
        ax = plt.subplot(2, 2, i+1)
        ax.plot(range(len(loss_type)), loss_type)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title("Train_"+key)
    plt.tight_layout()
    plt.savefig(f"./RESULT/loss/{name}loss.svg")
    plt.close()

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

    plt.savefig(f"./RESULT/scatter/{name}scatter.svg")
    plt.close()

    # record the valid_res
    with open(f"./RESULT/JSON/{name}.json", mode="w", encoding='utf-8') as f:
        json.dump(valid_res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 5*5
    for i in range(1, 6):
        for j in range(1, 6):
            pipline(
                initial_threshold=i, final_threshold=j, file_name="./data/清洗过的数据-7AA.csv",
                initial_index="plasmid_counts", final_index="start_virus_counts",
                seqs_index="realAAseq", fitness_index="fitness", input_dim=7 * 20, latent_dim=20,
                batch_size=32, lr=1e-3, max_epoch=100, patience=10
            )

