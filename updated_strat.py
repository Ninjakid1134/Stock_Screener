import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

import investing_functions
import ticker_lists

warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)

tickers = ticker_lists.all_tickers

# Dictionary to store results per sector

sector_results = {
    'technology': [],
    'consumer-cyclical': [],
    'communication-services': [],
    'healthcare': [],
    'consumer-defensive': [],
    'financial-services': [],
    'energy': [],
    'basic-materials': [],
    'industrials': [],
    'utilities': [],
    'real-estate': [],
}


# Define which metrics to calculate for each sector
# Each entry is (Label to show in results, function to calculate metric)
sector_metrics = {
    'technology': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("Revenue Growth", investing_functions.cagr_revenue),
        ("ROE", investing_functions.avg_roe),
        ("PEG", investing_functions.peg_ratio),
        ("FCF Margin", investing_functions.fcf_margin_average),
        ("Average R&D", investing_functions.r_d_ratio),
        ("ROIC", investing_functions.roic_average),
    ],
    'consumer-cyclical': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("Revenue Growth", investing_functions.cagr_revenue),
        ("EPS", investing_functions.cagr_eps),
        ("P/B", investing_functions.pb_ratio),
        ("D/E", investing_functions.debt_to_equity),
        ("Operating Margin", investing_functions.operating_margin),
        ("ROIC", investing_functions.roic_average),
    ],
    'communication-services': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("ROE", investing_functions.roe),
        ("PEG", investing_functions.peg_ratio),
        ("D/EBITDA", investing_functions.d_ebitda),
        ("FCF Yield", investing_functions.fcf_yield),
        ("Operating Margin", investing_functions.operating_margin),
        ("ROIC", investing_functions.roic_average),
    ],
    'healthcare': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("ROE", investing_functions.roe),
        ("P/E", investing_functions.pe_ratio),
        ("Operating Margin", investing_functions.operating_margin),
        ("Average R&D", investing_functions.r_d_ratio),
        ("ROIC", investing_functions.roic_average),
        ("Gross Margin", investing_functions.gross_margin),
    ],
    'consumer-defensive': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("FCF Margin", investing_functions.fcf_margin_average),
        ("EPS", investing_functions.cagr_eps),
        ("ROE", investing_functions.roe),
        ("Operating Margin", investing_functions.operating_margin),
        ("Dividend Payout Ratio", investing_functions.div_payout_ratio),
        ("Annual Dividends", investing_functions.div_annual),
    ],
    'financial-services': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("ROE", investing_functions.roe),
        ("ROA", investing_functions.roa),
        ("Revenue Growth", investing_functions.cagr_revenue),
        ("P/B", investing_functions.pb_ratio),
        ("PEG", investing_functions.peg_ratio),
        ("FCF", investing_functions.fcf_average),
        ("EPS", investing_functions.cagr_eps),
        ("D/E", investing_functions.debt_to_equity),
    ],
    'energy': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("FCF", investing_functions.fcf_average),
        ("D/E", investing_functions.debt_to_equity),
        ("ROCE", investing_functions.roce),
        ("EV/EBITDA", investing_functions.ev_ebitda_calc),
        ("Dividend Coverage Ratio", investing_functions.div_coverage_ratio),
    ],
    'basic-materials': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("EV/EBITDA", investing_functions.ev_ebitda_calc),
        ("D/EBITDA", investing_functions.d_ebitda),
        ("Gross Margin", investing_functions.gross_margin),
        ("ROIC", investing_functions.roic_average),
        ("Inventory", investing_functions.inventory_turnover_ratio),
    ],
    'industrials': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("ROA", investing_functions.roa),
        ("Operating Margin", investing_functions.operating_margin),
        ("D/E", investing_functions.debt_to_equity),
        ("ROIC", investing_functions.roic_average),
        ("EV/EBITDA", investing_functions.ev_ebitda_calc),
    ],
    'utilities': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("ROE", investing_functions.roe),
        ("Operating Margin", investing_functions.operating_margin),
        ("Dividend Payout Ratio", investing_functions.div_payout_ratio),
        ("D/E", investing_functions.debt_to_equity),
        ("Interest Cover Ratio", investing_functions.interest_cover_ratio),
    ],
    'real-estate': [
        ("Price", investing_functions.current_stock_price),
        ("Intrinsic Value", investing_functions.dcf_intrinsic_value),
        ("P/FFO", investing_functions.p_ffo),
        ("FFO GROWTH", investing_functions.ffo_cagr),
        ("D/E", investing_functions.debt_to_equity),
        ("Interest Cover Ratio", investing_functions.interest_cover_ratio),
        ("Dividend Payout Ratio (FFO)", investing_functions.dividend_pay_ffo),
    ],
}

# Loop through each ticker and process
for tick in tickers:
    try:
        ticker_obj = yf.Ticker(tick)  # create yfinance object
        sector = investing_functions.sector(tick)  # get sector from custom function

        # If sector is not in our defined metrics, skip
        if sector not in sector_metrics:
            continue

        # Fetch company name for display
        company_name = ticker_obj.info.get("longName", "N/A")
        data = {"Ticker": f'{tick} - {company_name}'}

        skip_stock = False  # flag to skip stock if data invalid
        # Loop through all metrics for this sector
        for label, func in sector_metrics[sector]:
            value = func(tick)  # calculate the metric using custom function
            # If value is None or negative (for numeric), skip this stock
            if value is None:
                skip_stock = True
                break
            if isinstance(value, (int, float)) and value < 0:
                print(f"Skipping {tick} due to negative value in {label}")
                skip_stock = True
                break
            data[label] = value  # store metric value in data dictionary

        if skip_stock:
            continue  # move to next ticker

        # Add industry info
        data["Industry"] = ticker_obj.info.get('industry', "N/A")
        price = data.get("Price")
        intrinsic = data.get("Intrinsic Value")

        # Validate price and intrinsic value
        if price is None or intrinsic is None:
            continue

        # Skip if intrinsic value is less than price (overvalued)
        if intrinsic < price:
            print(f"Skipping {tick} due to stock being overvalued")
            continue

        # Calculate Margin of Safety (MOS) = (Intrinsic - Price)/Price
        data["MOS"] = round((intrinsic - price) / price, 2) if price else None

        # Add this stock's data to sector results
        sector_results[sector].append(data)

    except (Exception, YFRateLimitError) as e:
        # Handle errors (e.g., rate limit, network errors)
        print(f"Error processing {tick}: {e}")
        time.sleep(5)  # wait before continuing
        continue

# Once all tickers processed, display results per sector
for sector, results in sector_results.items():
    print(f"\n{sector.upper()}:\n")
    df = pd.DataFrame(results)  # convert results to DataFrame
    if df.empty:
        print("No qualifying stocks.\n")
        continue

    try:
        # Fill missing values with NaN to avoid errors in filtering
        df = df.fillna(np.nan)

        # Apply sector-specific filtering rules:
        if sector == 'technology':
            df = df[
                (df["ROE"] >= 0.05) &
                (df['Revenue Growth'] >= 0.03) &
                (df['PEG'] <= 5)
            ]
        elif sector == 'consumer-cyclical':
            df = df[
                (df['Revenue Growth'] >= 0.03) &
                (df['EPS'] >= 0.02) &
                (df["P/B"] <= df['P/B'].quantile(0.90)) &
                (df['D/E'] <= 3)
            ]
        elif sector == 'communication-services':
            df = df[
                (df['ROE'] >= 0.03) &
                (df['PEG'] <= 5) &
                (df["Operating Margin"] >= 0.04)
            ]
        # ... similar filter logic for other sectors
        elif sector == 'real-estate':
            df = df[
                (df['D/E'] <= 3)
            ]
    except Exception as e:
        # Catch errors in filtering step
        print(f"Filtering error for {sector}: {e}")
        continue

    # Print final filtered DataFrame for this sector
    print(df.to_string(index=False))