import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load data
claim_data = pd.read_csv('claim_data_group5_2024.csv')

# Check for missing values (NA)
missing_values = claim_data.isnull().sum()
print("NA counts:\n", missing_values)

# Drop rows with NAs (or use other methods based on context)
claim_data = claim_data.dropna()

# Filter `ClaimNb` and `ClaimAmount` for non-negative values (remove anomalies)
claim_data = claim_data[(claim_data['ClaimNb'] >= 0) & (claim_data['ClaimAmount'] >= 0)]

# Estimate frequency distribution (Poisson)
claim_frequency = claim_data['ClaimNb']
lambda_est = claim_frequency.mean()

# Estimate severity distribution (Gamma)
claim_severity = claim_data['ClaimAmount']
alpha_est, loc_est, beta_est = stats.gamma.fit(claim_severity, floc=0)

# Simulate total losses for the next year
num_simulations = 10000
total_losses = []

for _ in range(num_simulations):
    # Simulate claim frequency
    simulated_claims = np.random.poisson(lambda_est)
    
    # Simulate claim severity
    simulated_severity = np.random.gamma(alpha_est, beta_est, simulated_claims)
    
    # Calculate total loss
    total_loss = simulated_severity.sum()
    total_losses.append(total_loss)

# Convert to numpy array for easier manipulation
total_losses = np.array(total_losses)

# Plot the distribution of total losses
plt.hist(total_losses, bins=50, density=True, alpha=0.6, color='g')
plt.title('Simulated Total Losses for Next Year')
plt.xlabel('Total Loss')
plt.ylabel('Frequency')
plt.show()

# Print basic stats for simulated total losses
print("Simulated Total Losses - Mean:", total_losses.mean())
print("Simulated Total Losses - Std Dev:", total_losses.std())

# Goodness-of-fit test for severity distribution
ks_stat, p_value = stats.kstest(claim_severity, 'gamma', args=(alpha_est, loc_est, beta_est))
print("KS Test for Gamma Distribution - Statistic:", ks_stat, "P-value:", p_value)

# You can add more goodness-of-fit tests and compare different distributions here