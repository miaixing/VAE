import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchmetrics
from torchmetrics.functional import r2_score, pearson_corrcoef
import matplotlib.pyplot as plt

# Define the dataset class
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


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=140, latent_dim=64):
        super(VAE, self).__init__()

        # Encoder
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean+eps * std

    def forward(self, x):
        # Encode input to latent space parameters
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterize(mean, logvar)
        # Decode latent representation to reconstruct input
        x_recon = self.decoder(z)
        return x_recon, mean, logvar, z


# Define the regression model
class FitnessRegressor(nn.Module):
    def __init__(self, input_dim=140, latent_dim=64):
        super(FitnessRegressor, self).__init__()
        self.vae = VAE(input_dim, latent_dim)

        # Fusion layer and regressor
        self.regressor = nn.Sequential(
            nn.Linear(input_dim+latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Get reconstructed input and latent features from VAE
        x_recon, mean, logvar, z = self.vae(x)
        # Concatenate original input and latent features
        fusion = torch.cat([x, z], dim=1)
        # Predict fitness
        fitness_pred = self.regressor(fusion)
        return x_recon, fitness_pred, mean, logvar


# Training process
def train_model(model, dataloader, epochs=50, lr=1e-3, beta=1.0):
    criterion_recon = nn.MSELoss()
    criterion_regress = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in range(epochs):
        model.train()
        total_loss, recon_loss, regress_loss, kl_loss = 0, 0, 0, 0

        for x, y in dataloader:
            x, y = x.float(), y.float()

            # Forward pass
            x_recon, y_pred, mean, logvar = model(x)

            # Compute reconstruction loss
            loss_recon = criterion_recon(x_recon, x)

            # Compute KL divergence
            kl_div = -0.5 * torch.sum(1+logvar-mean.pow(2)-logvar.exp(), dim=1).mean()

            # Compute regression loss
            loss_regress = criterion_regress(y_pred.squeeze(), y)

            # Total loss
            loss = loss_recon+beta * kl_div+loss_regress

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += loss_recon.item()
            regress_loss += loss_regress.item()
            kl_loss += kl_div.item()

        loss_list.append(regress_loss)
        print(
            f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}, Reconstruction Loss: {recon_loss:.4f}, "
            f"KL Loss: {kl_loss:.4f}, Regression Loss: {regress_loss:.4f}"
        )
        # print(loss_list)
        if len(loss_list) > 2 and loss_list[-1] > loss_list[-2]:
            print("converged")
            break
        else:
            print("Not converged")

def valid_model(model, dataloader, beta=1.0):
    model.eval()
    total_loss, recon_loss, regress_loss, kl_loss = 0, 0, 0, 0
    criterion_recon = nn.MSELoss()
    criterion_regress = nn.MSELoss()
    enrichment_true = None
    enrichment_pred = None

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.float(), y.float()

            # Forward pass
            x_recon, y_pred, mean, logvar = model(x)

            # Compute reconstruction loss
            loss_recon = criterion_recon(x_recon, x)

            # Compute KL divergence
            kl_div = -0.5 * torch.sum(1+logvar-mean.pow(2)-logvar.exp(), dim=1).mean()

            # Compute regression loss
            loss_regress = criterion_regress(y_pred.squeeze(), y)

            # Total loss
            loss = loss_recon+beta * kl_div+loss_regress

            total_loss += loss.item()
            recon_loss += loss_recon.item()
            regress_loss += loss_regress.item()
            kl_loss += kl_div.item()
            if enrichment_pred is None:
                enrichment_pred = y_pred
                enrichment_true = y
            else:
                enrichment_pred = torch.cat((enrichment_pred, y_pred), dim=0)
                enrichment_true = torch.cat((enrichment_true, y), dim=0)

        print(
            f"Total Loss: {total_loss:.4f}, Reconstruction Loss: {recon_loss:.4f}, "
            f"KL Loss: {kl_loss:.4f}, Regression Loss: {regress_loss:.4f}"
        )
        print(enrichment_true.shape, enrichment_pred.shape)
        r2 = r2_score(preds=enrichment_pred.view(-1), target=enrichment_true)
        pearson = pearson_corrcoef(preds=enrichment_pred.view(-1), target=enrichment_true)
        print(r2, pearson)

        # scatter
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        ax.scatter(enrichment_true.tolist(), enrichment_pred.tolist())
        plt.show()


# Example usage
if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer="./data/high.csv", index_col=None)
    df = df[df["InitialCount"] > 1]
    df = df[df["FinalCount"] > 1]
    seqs = df["AASeq"].to_list()
    enrichment = df["LogEnrichment"].to_list()
    dataset = AminoAcidDataset(seqs, enrichment)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Initialize and train the model
    model = FitnessRegressor(input_dim=17*20, latent_dim=32)
    train_model(model, dataloader, epochs=100, lr=1e-3)
    valid_model(model, dataloader)

