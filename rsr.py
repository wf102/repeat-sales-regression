# https://www.sciencedirect.com/science/article/pii/S1051137705800282

# Raw file downloaded from "https://landregistry.data.gov.uk/app/ppd/?relative_url_root=%2Fapp%2Fppd"
# Search for Postcode = GL51
# Download subsequent CSV as all fields for GL51 district (ppd_gl51.csv) - 29769 rows (as of May 2026)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from utils import process_sale_pairs, generate_design_matrix, create_plot, process_ppd, generate_sale_pairs

DIR = "data"
period = "Q"

# Read and process sales pairs
sale_pairs, date_values = process_sale_pairs(f"{DIR}/sale_pairs.csv", period)

# Generate vector of dependant variables y (ln(P2/P1))
y = np.array(sale_pairs["log_price_diff"].values)

# Generate design matrix
M = generate_design_matrix(sale_pairs, period)

# Remove first column to set first parameter as base index
M_red = M[:, 1:] 

# Perform least squares regression with SciPy to find index parameters
params, *_ = lstsq(M_red, y)

# Add in base parameter as 0
params = np.concatenate([[0], params])

# Exponentiate to recover House Price Index (HPI=1 in 1995)
hpi = np.exp(params)

# Generate plot
create_plot(date_values, hpi)
