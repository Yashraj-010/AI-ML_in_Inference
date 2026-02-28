from matplotlib.pylab import sample
import numpy as np
import torch 
import os
from tqdm import tqdm, trange
class MCMC():
    def __init__(self, 
                model, 
                obsvation_data,
                covariance,  
                likelihood = "loglikelihood", 
                prior: dict = {"type": "uniform", 
                                    "M": [10,800], 
                                    "N": [10,200], 
                                    "R": [2,40]}):
        self.model = model
        self.obsvation_data = obsvation_data
        self.covariance = covariance
        self.likelihood = likelihood
        self.prior = prior
        self.prior_type = prior["type"]
        self.prior_params = len(prior) - 1  

    def loglikelihood_ANN(self, params):
        # M, N, R = params
        model_prediction = self.model(params)
        residual = self.obsvation_data - model_prediction
        log_likelihood = -0.5 * np.dot(residual.T, np.linalg.solve(self.covariance, residual))
        return log_likelihood   


    def loglikelihood_BNN(self, params):
        # M, N, R = params
        model_mean, model_var = self.model(params)
        cov = self.covariance + model_var
        factor = np.log(np.linalg.det(cov)) + len(self.obsvation_data) * np.log(2 * np.pi)
        residual = self.obsvation_data - model_mean
        log_likelihood = -0.5 * (factor + np.dot(residual.T, np.linalg.solve(cov, residual)))
        return log_likelihood
    
    def run_mcmc(self, initial_params ,  proposal_std, num_samples=1000, model_type = "ANN"):
        samples = []
        accept = 0
        current_params = initial_params
        samples.append(current_params)
        current_log_likelihood = self.loglikelihood_BNN(current_params) if model_type == "BNN" else self.loglikelihood_ANN(current_params)

        for i in trange(num_samples):
            proposed_params = current_params + np.random.normal(0, proposal_std, size=current_params.shape)
            if (self.prior["M"][0]<=proposed_params[0]<=self.prior["M"][1] and 
                self.prior["N"][0]<=proposed_params[1]<=self.prior["N"][1] and 
                self.prior["R"][0]<=proposed_params[2]<=self.prior["R"][1]):
                proposed_log_likelihood = self.loglikelihood_ANN(proposed_params) if model_type == "ANN" else self.loglikelihood_BNN(proposed_params)
            else:
                proposed_log_likelihood = -np.inf 

        
            
            acceptance_prob = min(0, proposed_log_likelihood - current_log_likelihood)
            if np.log(np.random.rand()) < acceptance_prob:
                current_params = proposed_params
                current_log_likelihood = proposed_log_likelihood
                accept += 1
            
            samples.append(current_params)
        
        print(f"Total Samples: {num_samples}")
        print(f"Acceptance Ratio: {100 * accept/num_samples:.2f}")

        return np.array(samples)

