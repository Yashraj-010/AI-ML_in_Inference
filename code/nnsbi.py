from sbi.neural_nets import posterior_nn
from sbi.inference import SNPE, SNLE, SNRE, DirectPosterior, FMPE, NPE
from sbi.diagnostics import sbc
from sbi.utils import BoxUniform
import torch
import numpy as np

# Developing different SBI training and inference methods

class SBI:
    def __init__(self, training_data, prior, method='FMPE', num_simulations=1000):
        self.theta = torch.tensor(training_data[0], dtype=torch.float32)
        self.x = torch.tensor(training_data[1], dtype=torch.float32)
        self.prior = BoxUniform(low=prior[0], high=prior[1])
        self.method = method
        self.num_simulations = num_simulations
        self.trainer = None

    def train(self):
        if self.method == 'SNPE':
            self.trainer = SNPE(prior=self.prior)
        elif self.method == 'SNLE':
            self.trainer = SNLE(prior=self.prior)
        elif self.method == 'SNRE':
            self.trainer = SNRE(prior=self.prior)
        elif self.method == 'DirectPosterior':
            self.trainer = DirectPosterior(prior=self.prior)
        elif self.method == 'FMPE':
            self.trainer = FMPE(prior=self.prior)
        elif self.method == 'NPE':
            self.trainer = NPE(prior=self.prior)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # # if you use simulator
        # theta, x = [], []
        # for _ in range(self.num_simulations):
        #     theta_i = self.prior.sample()
        #     x_i = self.simulator(theta_i)
        #     theta.append(theta_i)
        #     x.append(x_i)

        # theta = np.array(theta)
        # x = np.array(x)

        # Train the inference method
        self.trainer.append_simulations(self.theta, self.x)
        self.trainer.train()
        self.posterior = self.trainer.build_posterior()
        return self.posterior

    def infer(self, observation, samples=10000):
        # if not self.trainer:
        #     raise ValueError("Inference method not trained yet.")
        observation = torch.tensor(observation, dtype=torch.float32)
        
        posterior_samples = self.posterior.sample(samples, x=observation)
        return posterior_samples