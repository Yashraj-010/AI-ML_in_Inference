import numpy as np
import torch 
import os

class MCMC():
    def __init__(self, model, obsvation_data, covariance,  likelihood = "loglikelihood", ):
        self.model = model
        self.obsvation_data = obsvation_data
        self.covariance = covariance
        self.likelihood = likelihood