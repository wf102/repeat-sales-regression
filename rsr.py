# https://www.sciencedirect.com/science/article/pii/S1051137705800282

# Raw file downloadded from "https://landregistry.data.gov.uk/app/ppd/?relative_url_root=%2Fapp%2Fppd"
# Search for Postcode = GL51
# Download subsequent CSV as all fields for GL51 district (ppd_gl51.csv) - 29769 rows (as of May 2026)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from utils import get_sale_pairs, process_sale_pairs, generate_design_matrix, create_plot

DIR = "data"
period = "Q"

df_ppd = pd.read_csv(f"{DIR}/ppd_gl51_clean.csv")


# Generate repeat sales pairs and save to intermediate file
sale_pairs = get_sale_pairs(df_ppd)
sale_pairs.to_csv(f"{DIR}/sale_pairs.csv", index=False)

# Process sales pairs
sale_pairs, date_values = process_sale_pairs(sale_pairs, period)

# Generate vector of dependant variables y (ln(P2/P1))
sale_pairs["log_price_diff"] = np.log(sale_pairs["Price2"]) - np.log(sale_pairs["Price1"])
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
