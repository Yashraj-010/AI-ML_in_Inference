import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormalizingFlow
from pyro.distributions.transforms import affine_autoregressive

import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pyro.enable_validation(False)
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True

class Data(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class BayesianNN(PyroModule):

    def __init__(self, input_dim=3, hidden=[64, 64], output_dim=7,
                 prior_scale=0.5, device="cpu"):

        super().__init__()

        self.device = device
        self.hidden_dims = hidden

        self.layers = PyroModule[nn.ModuleList]([])

        prev_dim = input_dim

        for h_dim in hidden:

            layer = PyroModule[nn.Linear](prev_dim, h_dim)

            weight_loc = torch.zeros(h_dim, prev_dim, device=device)
            weight_scale = torch.full((h_dim, prev_dim), prior_scale, device=device)

            bias_loc = torch.zeros(h_dim, device=device)
            bias_scale = torch.full((h_dim,), prior_scale, device=device)

            layer.weight = PyroSample(
                dist.Normal(weight_loc, weight_scale).to_event(2)
            )

            layer.bias = PyroSample(
                dist.Normal(bias_loc, bias_scale).to_event(1)
            )

            self.layers.append(layer)
            prev_dim = h_dim

        self.out = PyroModule[nn.Linear](prev_dim, output_dim)

        w_loc = torch.zeros(output_dim, prev_dim, device=device)
        w_scale = torch.full((output_dim, prev_dim), prior_scale, device=device)

        b_loc = torch.zeros(output_dim, device=device)
        b_scale = torch.full((output_dim,), prior_scale, device=device)

        self.out.weight = PyroSample(
            dist.Normal(w_loc, w_scale).to_event(2)
        )

        self.out.bias = PyroSample(
            dist.Normal(b_loc, b_scale).to_event(1)
        )

        # Observation noise
        self.sigma = PyroSample(
            dist.Uniform(
                torch.tensor(0., device=device),
                torch.tensor(1., device=device)
            )
        )

    def forward(self, x, y=None):

        for layer in self.layers:
            x = torch.relu(layer(x))

        mean = self.out(x)

        sigma = self.sigma

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mean, sigma).to_event(1),
                obs=y
            )

        return mean

class MCMCTrain():

        def __init__(self, hidden_layers, training_data, num_samples=1000, warmup_steps=200, path = "MCMC_BNN_model" ):
            # self.model = model
            self.hidden_layers = hidden_layers

            self.model = BayesianNN(hidden=self.hidden_layers, device=device)
            self.num_samples = num_samples
            self.warmup_steps = warmup_steps
            self.training_input = torch.from_numpy(training_data[0]).float().to(device)
            self.training_target = torch.from_numpy(training_data[1]).float().to(device)
            self.path = path
    
        def train(self,):
            nuts_kernel = NUTS(self.model, jit_compile=True)
            mcmc = MCMC(nuts_kernel, num_samples=self.num_samples, warmup_steps=self.warmup_steps)
            print("Starting MCMC sampling...")
            mcmc.run(self.training_input, self.training_target)
            samples = mcmc.get_samples()
            print("MCMC sampling completed.")
            print("Saving the model...")
            self.save_mcmc(samples)

            return samples

        def save_mcmc(self, samples):
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            checkpoint = {
            "method": "MCMC",
            "hidden_layers": self.hidden_layers,
            "num_samples": self.num_samples,
            "warmup_steps": self.warmup_steps,
            "posterior_samples": samples
            }
            
            filename = self.path + "/best_mcmc_model.pth"

            torch.save(checkpoint, filename)
            print("Best MCMC model saved at: ", filename)

class SVITrain():

        # def __init__(self, training_data, hidden_layers, batch_size=32, epochs=100, learning_rate=0.001, patience = 10 , path = "SVI_model" ):
        #     self.training_data = training_data
        #     # self.training_target = training_data[1].to(device)
        #     self.hidden_layers = hidden_layers
        #     self.batch_size = batch_size
        #     self.epochs = epochs
        #     self.learning_rate = learning_rate
        #     self.patience = patience
        #     self.path = path

        #     cpu_model = BayesianNN(hidden=self.hidden_layers, device="cpu")
        #     self.guide = AutoNormalizingFlow(
        #         cpu_model,
        #         lambda latent_dim: affine_autoregressive(
        #             latent_dim, 
        #             hidden_dims=[latent_dim, latent_dim]
        #         )
        #     )

        #     # 2. Run a dummy forward pass with a single CPU data point to FORCE lazy initialization
        #     x_dummy = torch.tensor(self.training_data[0][:1], dtype=torch.float32)
        #     y_dummy = torch.tensor(self.training_data[1][:1], dtype=torch.float32)
        #     self.guide(x_dummy, y_dummy)

        #     self.guide.to(device)
        #     pyro.get_param_store().to(device)

        #     # 4. Initialize the REAL model directly on the GPU and re-link the guide
        #     self.model = BayesianNN(hidden=self.hidden_layers, device=device).to(device)
        #     self.guide.model = self.model
            
        #     self.optimizer = Adam({"lr": self.learning_rate})
        #     self.svi = SVI(
        #         self.model, 
        #         self.guide, 
        #         self.optimizer, 
        #         loss=Trace_ELBO(num_particles=5, vectorize_particles=True)
        #     )


        #     self.model = BayesianNN(
        #                             hidden=self.hidden_layers,
        #                             device=device
        #                             ).to(device)

        #     # # self.model = BayesianNN(hidden=self.hidden_layers).to(device)
        #     # print("Moving model to device: ", device)
        #     # all_on_cuda = all(p.is_cuda for p in self.model.parameters())
        #     # print(f"Are all model parameters on CUDA? {all_on_cuda}")
            
        #     # # self.guide = AutoNormalizingFlow(self.model,lambda latent: affine_autoregressive(latent, hidden_dims=self.hidden_layers)).to(device)
        #     # self.guide = AutoNormalizingFlow(
        #     #     self.model,
        #     #     lambda latent_dim: affine_autoregressive(
        #     #         latent_dim, 
        #     #         hidden_dims=[latent_dim, latent_dim]
        #     #     )
        #     # ).to(device)
        #     # print("Moving guide to device: ", device)
        #     # all_on_cuda = all(p.is_cuda for p in self.guide.parameters())
        #     # print(f"Are all guide parameters on CUDA? {all_on_cuda}")

        #     # # pyro.clear_param_store()

        #     # # pyro.get_param_store().to(device)
            
        #     # self.optimizer = Adam({"lr": self.learning_rate})
        #     # # pyro.clear_param_store()
        #     # self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO(num_particles=5, vectorize_particles=True))
        #     # # all_on_cuda = all(p.is_cuda for p in self.svi.parameters())
        #     # # print(f"Are all svi parameters on CUDA? {all_on_cuda}")

        def __init__(self, training_data, hidden_layers, batch_size=32, epochs=100, learning_rate=0.001, patience=10, path="BNN_SVI_model"):
            self.training_data = training_data
            self.hidden_layers = hidden_layers
            self.batch_size = batch_size
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.patience = patience
            self.path = path

            pyro.clear_param_store()

            # 1. Initialize the model securely on the GPU
            self.model = BayesianNN(hidden=self.hidden_layers, device=device).to(device)
            
            # 2. Initialize the guide lazily
            self.guide = AutoNormalizingFlow(
                self.model,
                lambda latent_dim: affine_autoregressive(
                    latent_dim, 
                    hidden_dims=[latent_dim, latent_dim]
                )
            )

            # 3. Trigger the lazy initialization with a dummy pass.
            # (The AutoGuide will spawn its networks on the CPU initially)
            x_dummy = torch.tensor(self.training_data[0][:1], dtype=torch.float32).to(device)
            y_dummy = torch.tensor(self.training_data[1][:1], dtype=torch.float32).to(device)
            self.guide(x_dummy, y_dummy)

            # 4. THE FIX: Move the guide to the device AFTER the dummy pass.
            # This pushes the initialized networks to CUDA *and* automatically syncs the ParamStore.
            self.guide.to(device)
            
            self.optimizer = Adam({"lr": self.learning_rate})
            self.svi = SVI(
                self.model, 
                self.guide, 
                self.optimizer, 
                loss=Trace_ELBO(num_particles=5, vectorize_particles=False)
            )


        def train(self,):
            train_losses = []

            train_loader = DataLoader(Data(self.training_data[0], self.training_data[1]), batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.epochs):

                epoch_loss = 0

                for x, y in train_loader:
                    x = x.to(device, non_blocking=True)
                    print("x device: ", x.device)
                    y = y.to(device, non_blocking=True)
                    print("y device: ", y.device)
                    loss = self.svi.step(x, y)
                    # train_losses.append(loss)
                    
                    epoch_loss += loss

                if epoch % 10 == 0:
                    print(f"Epoch {epoch} - Loss: {loss:.6f}")

                epoch_loss /= len(train_loader)

                train_losses.append(epoch_loss)
            
            self.save_svi(train_losses)

            return train_losses, self.model, self.guide

        def save_svi(self, losses):
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            pyro.get_param_store().save(self.path + "/best_svi_model.params")

            checkpoint = {
            "method": "SVI",
            "hidden_layers": self.hidden_layers,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "train_losses": losses,
            "param_store": self.path + "/best_svi_model.params"
            }
            
            filename = self.path + "/best_svi_model.pth"

            torch.save(checkpoint, filename)
            print("Best SVI model saved at: ", filename)

class TrainBNN():
    """
    This class will train the BNN using either MCMC or SVI methods depending on the user input.

    """

    def __init__(self, method, training_data, hidden_layers, num_samples=1000, warmup_steps=200, batch_size=32, epochs=100, learning_rate=0.001, patience=10, path = "BNN_model" ):
        self.method = method
        self.training_data = training_data
        self.hidden_layers = hidden_layers
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.path = path

    def train(self,):
        if self.method == "MCMC":
            # model = BayesianNN(hidden=self.hidden_layers).to(device)
            mcmc_trainer = MCMCTrain(hidden_layers=self.hidden_layers, 
                                    training_data=self.training_data, 
                                    num_samples=self.num_samples, 
                                    warmup_steps=self.warmup_steps,
                                    path=self.path,
                                    )
            return mcmc_trainer.train()

        elif self.method == "SVI":
            svi_trainer = SVITrain(training_data=self.training_data, 
                                   hidden_layers=self.hidden_layers, 
                                   batch_size=self.batch_size, 
                                   epochs=self.epochs, 
                                   learning_rate=self.learning_rate, 
                                   patience=self.patience, 
                                   path=self.path,
                                   )
            return svi_trainer.train()


class BNNPredict():
    """
    Unified predictor for SVI and MCMC trained Bayesian Neural Networks
    """

    def __init__(self, model_path, device=None):

        self.model_path = model_path

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.checkpoint = torch.load(model_path, map_location=self.device)

        self.method = self.checkpoint["method"]

        self.model = None
        self.guide = None
        self.posterior_samples = None

        self._load_model()

    # ============================
    # Load model depending on method
    # ============================

    def _load_model(self):

        hidden_layers = self.checkpoint["hidden_layers"]

        self.model = BayesianNN(hidden=hidden_layers).to(self.device)

        if self.method == "SVI":

            self.guide = AutoNormalizingFlow(
                self.model,
                lambda latent: affine_autoregressive(
                    latent,
                    hidden_dims=[64, 64]
                )
            )

            pyro.get_param_store().load(self.checkpoint["param_store"])

        elif self.method == "MCMC":

            self.posterior_samples = self.checkpoint["posterior_samples"]

        else:
            raise ValueError("Unknown method in checkpoint")

    # ============================
    # Prediction
    # ============================

    def predict(self, X, num_samples=200, return_numpy=True):

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.method == "SVI":

            predictive = Predictive(
                self.model,
                guide=self.guide,
                num_samples=num_samples,
                return_sites=("obs", "_RETURN")
            )

        else:  # MCMC

            predictive = Predictive(
                self.model,
                posterior_samples=self.posterior_samples,
                return_sites=("obs", "_RETURN")
            )

        samples = predictive(X)

        preds = samples["obs"]

        mean = preds.mean(0)
        std = preds.std(0)

        if return_numpy:
            mean = mean.detach().cpu().numpy()
            std = std.detach().cpu().numpy()

        return mean, std