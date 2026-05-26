import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def process_ppd(file_in, file_out):
    
    """
    Reads a raw PPD CSV from eg https://landregistry.data.gov.uk/app/ppd/?relative_url_root=%2Fapp%2Fppd
    Removes unecessary fields, saves to CSV file.
    """

    columns = {"Price": 1, "Date": 2, "Postcode": 3, "Paon": 7, "Saon": 8}

    df = pd.read_csv(file_in, usecols=columns.values(), header=None)
    df.columns = columns.keys()

    # Generate unique identifier for each property based on Paon, Saon, and Postcode
    df["id"] = (
        df["Paon"].fillna("").str.strip() + "|" +
        df["Saon"].fillna("").str.strip() + "|" +
        df["Postcode"].fillna("").str.strip()
    )

    # Factorize the 'id' column to convert it into numeric values
    df["id"], _ = pd.factorize(df["id"])

    df[["Price", "Date", "id"]].to_csv(file_out, index=False)


def generate_sale_pairs(file_in, file_out):

    """Reads in the cleaned PPD, generates raw sale pairs data and saves to CSV."""

    df = pd.read_csv(file_in)

    # Group by 'id' and sort by 'Date' within each group
    df_sorted = df.sort_values(by=['id', 'Date'])
    
    # Create a new DataFrame to hold the sale pairs
    sale_pairs = pd.DataFrame(columns=['id', 'Date1', 'Date2', 'Price1', 'Price2'])
    
    # Iterate through each group of properties
    for property_id, group in df_sorted.groupby('id'):

        if len(group) > 1:

            n_repeat_sales = len(group) - 1

            for i in range(n_repeat_sales):

                first_sale = group.iloc[i]
                second_sale = group.iloc[i + 1]
                
                sale_pairs = pd.concat([sale_pairs, pd.DataFrame([{
                    'id': property_id,
                    'Date1': first_sale['Date'],
                    'Date2': second_sale['Date'],
                    'Price1': int(first_sale['Price']),
                    'Price2': second_sale['Price'],
                }])], ignore_index=True)

    sale_pairs.to_csv(file_out, index=False)


def process_sale_pairs(sale_pairs, period):

    """
    Processes sale pairs:
        - Rounds all sale dates to nearest time period
        - Calculates log price difference for each sale pair
        - Removes sales pairs with both dates within the same period
    Calculate date values for each time period.      
    """

    sale_pairs = pd.read_csv("data/sale_pairs.csv", dtype={
        'id': 'int64',
        'Date1': 'string',
        'Date2': 'string',
        'Price1': 'int64',
        'Price2': 'int64',
    })

    # Round dates to nearest time period (ie month, quarter, year...)
    sale_pairs["Date1"] = pd.to_datetime(sale_pairs["Date1"]).dt.to_period(period)
    sale_pairs["Date2"] = pd.to_datetime(sale_pairs["Date2"]).dt.to_period(period)

    # Remove pairs where both sales are in the same period
    sale_pairs = sale_pairs[sale_pairs["Date1"] != sale_pairs["Date2"]].reset_index() 

    # Generate log price difference (dependent variable for regression)
    sale_pairs["log_price_diff"] = np.log(sale_pairs["Price2"]) - np.log(sale_pairs["Price1"])

    dates = np.sort(list(sale_pairs[["Date1","Date2"]].stack().unique()))
    date_values = pd.to_datetime(pd.PeriodIndex(dates, freq=period).to_timestamp())

    return sale_pairs, date_values


def generate_design_matrix(sale_pairs, period):

    """
    Generates design matrix for repeat sales regression.
    Matrix is zeros, -1 for the first sale, +1 for the second sale.
    This allows the matrix to pick out the relevant index value for each transaction.
    """

    n_pairs = len(sale_pairs)
    dates = np.sort(list(sale_pairs[["Date1","Date2"]].stack().unique()))
    n_dates = len(dates)

    date_values = pd.to_datetime(pd.PeriodIndex(dates, freq=period).to_timestamp())

    sale_pairs["Date1"] = sale_pairs["Date1"].astype('string')
    sale_pairs["Date2"] = sale_pairs["Date2"].astype('string')

    sale_pairs["Date1_index"] = sale_pairs["Date1"].map(dict(zip([str(d) for d in dates], range(len(dates)))))
    sale_pairs["Date2_index"] = sale_pairs["Date2"].map(dict(zip([str(d) for d in dates], range(len(dates)))))

    M = np.zeros((n_pairs, n_dates))

    # Populate with dummy variables (-1 for Date1, +1 for Date2)
    for i, row in sale_pairs.iterrows():
        M[i, row["Date1_index"]] = -1
        M[i, row["Date2_index"]] = 1

    return M


def create_plot(date_values, hpi):

    """Plots index, saves to PNG."""

    sns.set_theme()

    fig, ax = plt.subplots(figsize=(12,8))

    plt.plot(date_values, hpi)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.ylim((0, 1.1 * max(hpi)))
    plt.xlabel("Time")
    plt.ylabel("Index")
    plt.title("Repeat Sales Regression Price Index")
    plt.tight_layout()
    plt.savefig("hpi_plot.png")

