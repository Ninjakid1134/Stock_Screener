import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.stats import norm, skew, kurtosis
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.printoptions(suppress=True)

def get_fcf(ticker):
    """
    Fetch Free Cash Flow (FCF) for a given ticker.
    """
    stock = yf.Ticker(ticker)
    try:
        fcf = stock.cashflow.loc['Free Cash Flow'].dropna()
        return fcf
    except KeyError:  # 'Free Cash Flow' row not found
        return pd.Series(dtype=float)
    except AttributeError:  # cashflow attribute might not exist
        return pd.Series(dtype=float)

def get_cagr_fcf(ticker):
    """
    Calculate the CAGR of Free Cash Flow.
    Returns 0.0 if insufficient data or invalid values.
    """
    try:
        fcf = get_fcf(ticker)
        # Check that fcf is a pandas Series with at least 2 points
        if fcf is None or len(fcf) < 2:
            return 0.0

        # First and last values for CAGR
        start_fcf = fcf.iloc[0]
        end_fcf = fcf.iloc[-1]

        # Check for valid positive start value (cannot divide by zero or negative for CAGR)
        if start_fcf <= 0 or end_fcf <= 0:
            return 0.0

        years = len(fcf) - 1
        cagr_fcf = (end_fcf / start_fcf) ** (1 / years) - 1
        return round(cagr_fcf, 3)

    except Exception as e:
        print(f"Unexpected error in get_cagr_fcf for {ticker}: {e}")
        return 0.0

def project_fcf(ticker):
    """
    Project future FCF based on historical CAGR.
    """
    fcf = get_fcf(ticker)
    if fcf.empty or fcf.iloc[0] == 0:
        return []
    years = len(fcf)
    cagr = get_cagr_fcf(ticker)
    base_fcf = fcf.iloc[0]

    projections = []
    for i in range(1, years + 1):
        projected_value = base_fcf * ((1 + cagr) ** i)
        projections.append(round(projected_value, 0))

    return projections

def market_return(stocks):
    """
    Calculate annualised market returns for given stock(s).
    """
    if isinstance(stocks, str):
        stocks = [stocks]

    results = []

    for stock in stocks:
        try:
            data = yf.download(stock, interval="1mo")["Close"].pct_change().dropna()
            if data.empty:
                results.append(0.0)
                continue
            obs = len(data)
            compound_growth = (1 + data).prod()
            annualised_return = (compound_growth ** (12 / obs)) - 1
            results.append(round(annualised_return, 4))
        except (KeyError, ValueError) as e:
            print(f"Data error downloading or processing {stock}: {e}")
            results.append(0.0)
        except Exception as e:  # unexpected error fallback
            print(f"Unexpected error in market_return for {stock}: {e}")
            results.append(0.0)

    return pd.DataFrame(results, columns=["Annualised Return"], index=stocks)

def annualised_returns(stocks, n_periods, year):
    """
    Calculate annualised returns for given stock(s) over specified periods starting from 'year'.
    """
    if isinstance(stocks, str):
        stocks = [stocks]

    results = []

    for stock in stocks:
        try:
            data = yf.download(stock, interval="1mo")["Close"].pct_change().dropna()
            data = data.loc[f"{year}":]
            if data.empty:
                results.append((stock, 0.0))
                continue
            obs = len(data)
            compound_growth = (1 + data).prod()
            annualised_return = (compound_growth ** (n_periods / obs)) - 1
            results.append((stock, round(annualised_return, 4)))
        except (KeyError, ValueError) as e:
            print(f"Data error processing {stock}: {e}")
            results.append((stock, 0.0))
        except Exception as e:  # unexpected error fallback
            print(f"Unexpected error in annualised_returns for {stock}: {e}")
            results.append((stock, 0.0))

    return pd.DataFrame(results, columns=["Stock", "Annualised Return"])




def cornish_var(asset, level=5, year=2020, modified=False):
    """
    Calculate the Cornish-Fisher Value at Risk (VaR) for one or multiple assets.

    Parameters
    ----------
    asset : str or list of str
        A single ticker or list of ticker symbols.
    level : float, optional
        Confidence level (in percent) for VaR calculation. Default is 5.
    year : int or str, optional
        The starting year for data selection. Default is 2020.
    modified : bool, optional
        Placeholder parameter for future modifications. Default is False.

    Returns
    -------
    None
        Prints a pandas DataFrame showing the VaR for each asset and the average VaR.
    """
    if isinstance(asset, str):
        asset = [asset]
    o = pd.DataFrame()
    start = str(year) + "-1-1"
    for a in asset:
        my_data = yf.download(a, start=start)['Adj Close'].resample('M').ffill()
        log_rets = np.log(my_data/my_data.shift(1)).dropna()
        z = (level/100)
        S = skew(log_rets)
        K = kurtosis(log_rets, fisher=True)
        z_mod = z + ((z**2 - 1)*S/6) + ((z**3-3*z)*(K-3)/24) - ((2*z**3-5*z)*(S**2)/36)
        var_cornish = -(np.mean(log_rets) + z_mod*np.std(log_rets))
        o[a] = pd.Series([var_cornish], index=["Cornish-Fisher VaR"])
    o.loc["Average VaR"] = o.mean(axis=1)
    print(o)

def yearly_stock_returns(tickers, year):
    """
    Retrieve monthly percentage returns for given tickers from a specific year.

    Parameters
    ----------
    tickers : list of str
        List of ticker symbols.
    year : int or str
        Year from which to start the data.

    Returns
    -------
    pd.DataFrame
        DataFrame with monthly returns (percentage change) for each ticker.
    """
    start = str(year) + "-1-1"
    data = pd.DataFrame()
    for t in tickers:
        data[t] = yf.download(t, start=start)['Adj Close'].resample('M').ffill()
    return data.pct_change().dropna()

def portfolio_return(ticker, year):
    """
    Compute the equal-weighted portfolio return for a given set of tickers.

    Parameters
    ----------
    ticker : str or list of str
        A single ticker or list of tickers.
    year : int or str
        Year from which to start the calculation.

    Returns
    -------
    float
        Equal-weighted portfolio return (annualised).
    """
    if isinstance(ticker, str):
        ticker = [ticker]
    data = yearly_stock_returns(ticker, year)
    return data.mean(axis=1).mean()

def portfolio_volatility(ticker, year):
    """
    Compute equal-weighted portfolio volatility (standard deviation).

    Parameters
    ----------
    ticker : str or list of str
        A single ticker or list of tickers.
    year : int or str
        Year from which to start the calculation.

    Returns
    -------
    float
        Annualised portfolio volatility.
    """
    if isinstance(ticker, str):
        ticker = [ticker]
    data = yearly_stock_returns(ticker, year)
    return data.mean(axis=1).std() * np.sqrt(12)

def fcf_average(ticker):
    """
    Compute the average Free Cash Flow to Operating Cash Flow ratio.

    Parameters
    ----------
    ticker : dict
        A dictionary-like object containing at least 'freeCashflow' and 'operatingCashflow'.

    Returns
    -------
    float
        Free Cash Flow divided by Operating Cash Flow.
        Returns 0.0 if data is missing or invalid.
    """
    try:
        data = ticker
        return float(data.get("freeCashflow", 0)) / float(data.get("operatingCashflow", 1))
    except (TypeError, ZeroDivisionError, ValueError):
        return 0.0


def pe_ratio(ticker):
    """
    Compute the Price-to-Earnings (P/E) ratio.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Trailing P/E ratio from Yahoo Finance.
        Returns 0.0 if unavailable.
    """
    try:
        return yf.Ticker(ticker).info.get("trailingPE", 0.0)
    except (KeyError, AttributeError, ValueError):
        return 0.0


def pb_ratio(ticker):
    """
    Compute the Price-to-Book (P/B) ratio.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Price-to-Book ratio from Yahoo Finance.
        Returns 0.0 if unavailable.
    """
    try:
        return yf.Ticker(ticker).info.get("priceToBook", 0.0)
    except (KeyError, AttributeError, ValueError):
        return 0.0


def cagr_eps(ticker):
    """
    Compute the Compound Annual Growth Rate (CAGR) of basic earnings per share.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        CAGR of EPS based on Yahoo Finance earnings data.
        Returns 0.0 if insufficient or invalid data.
    """
    try:
        hist = yf.Ticker(ticker).earnings
        if hist.shape[0] < 2:
            return 0.0
        first = hist['Earnings'].iloc[0]
        last = hist['Earnings'].iloc[-1]
        n = hist.shape[0] - 1
        return (last / first) ** (1 / n) - 1
    except (KeyError, IndexError, ZeroDivisionError, ValueError):
        return 0.0


def beta_info(ticker):
    """
    Scrape and return the Beta value from AlphaQuery.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Beta value (0.0 if unavailable or error occurs).
    """
    try:
        url = f"https://www.alphaquery.com/stock/{ticker}/volatility-analysis/beta"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        beta_value = soup.find('div', {'class': 'value'}).text.strip()
        return float(beta_value)
    except (AttributeError, ValueError, requests.exceptions.RequestException):
        return 0.0


def ebit_info(ticker):
    """
    Scrape and return the most recent EBIT (Operating Income) from AlphaQuery.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        EBIT value (0.0 if unavailable or error occurs).
    """
    try:
        url = f"https://www.alphaquery.com/stock/{ticker}/financial-statements/income-statement"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        ebit_value = soup.find('td', text='EBIT').find_next('td').text.strip().replace(',', '')
        return float(ebit_value)
    except (AttributeError, ValueError, requests.exceptions.RequestException):
        return 0.0


def cagr_revenue(ticker):
    """
    Compute the Compound Annual Growth Rate (CAGR) of total revenue.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        CAGR of revenue (0.0 if insufficient data or error occurs).
    """
    try:
        hist = yf.Ticker(ticker).financials.loc['Total Revenue']
        if len(hist) < 2:
            return 0.0
        first = hist.iloc[-1]
        last = hist.iloc[0]
        n = len(hist) - 1
        return (last / first) ** (1 / n) - 1
    except (KeyError, IndexError, ZeroDivisionError, ValueError):
        return 0.0


def roic_average(ticker):
    """
    Return the Return on Equity (proxy for ROIC in this context).

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Return on equity value (0.0 if unavailable).
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("returnOnEquity", 0.0)
    except (KeyError, AttributeError):
        return 0.0


def div_annual(ticker):
    """
    Compute the total dividends paid in 2024.

    Parameters
    ----------
    ticker : str
        Ticker symbol.

    Returns
    -------
    float
        Total dividend amount in 2024 (0.0 if unavailable).
    """
    try:
        divs = yf.Ticker(ticker).dividends
        return float(divs[divs.index.year == 2024].sum())
    except (KeyError, AttributeError, ValueError):
        return 0.0


def mu(tickers, year):
    """
    Compute the annualized mean of monthly log returns.

    Parameters
    ----------
    tickers : str or list of str
        Single ticker or a list of tickers.
    year : int or str
        Starting year for calculation.

    Returns
    -------
    float
        Annualized mean log return (0.0 if error occurs).
    """
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        data = yearly_stock_returns(tickers, year)  # assumes you have this function
        log_ret = np.log(1 + data)
        return log_ret.mean(axis=1).mean() * 12
    except (KeyError, ValueError, AttributeError):
        return 0.0


def current_stock_price(tickers):
    """
    Fetch the most recent adjusted closing price.

    Parameters
    ----------
    tickers : str
        Ticker symbol.

    Returns
    -------
    float
        Latest stock price (0.0 if unavailable).
    """
    try:
        return yf.download(tickers, period='1d')['Adj Close'].iloc[-1]
    except (KeyError, IndexError, ValueError):
        return 0.0


def debt_to_equity(tickers):
    """
    Compute the Debt-to-Equity ratio.

    Parameters
    ----------
    tickers : str
        Ticker symbol.

    Returns
    -------
    float
        Debt-to-Equity ratio (0.0 if unavailable).
    """
    try:
        info = yf.Ticker(tickers).info
        return info.get("debtToEquity", 0.0)
    except (KeyError, AttributeError):
        return 0.0


def roe(tickers):
    """
    Compute the Return on Equity (ROE).

    Parameters
    ----------
    tickers : str
        Ticker symbol.

    Returns
    -------
    float
        ROE value (0.0 if unavailable).
    """
    try:
        info = yf.Ticker(tickers).info
        return info.get("returnOnEquity", 0.0)
    except (KeyError, AttributeError):
        return 0.0

def avg_roe(tickers):
    """
    Calculate the average Return on Equity (ROE) for a given ticker.

    Args:
        tickers (str): Stock ticker symbol.

    Returns:
        float: Average ROE, rounded to 2 decimal places. Returns 0.0 if data is unavailable.
    """
    try:
        net_income = yf.Ticker(tickers).financials.loc['Net Income'].iloc[0]
        shareholders_equity = yf.Ticker(tickers).balancesheet.loc['Stockholders Equity'].dropna()
        if len(shareholders_equity) < 2:
            return 0.0
        avg_equity = (shareholders_equity.iloc[-1] + shareholders_equity.iloc[0]) / 2
        if avg_equity == 0:
            return 0.0
        roes = net_income / avg_equity
        return round(roes, 2)
    except KeyError as e:
        print(f"Missing data for avg ROE for {tickers}: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error in avg_roe for {tickers}: {e}")
        return 0.0


def peg_ratio(symbols):
    """
    Calculate the PEG (Price/Earnings to Growth) ratio for a given symbol.

    Args:
        symbols (str): Stock ticker symbol.

    Returns:
        float: PEG ratio, rounded to 2 decimal places. Returns 0.0 if data is unavailable.
    """
    try:
        pe = pe_ratio(symbols)
        eps = cagr_eps(symbols)
        if eps == 0:
            return 0.0
        peg = pe / eps
        return round(peg, 2)
    except Exception as e:
        print(f"Unexpected error in peg_ratio for {symbols}: {e}")
        return 0.0


def sector(tickers):
    """
    Fetch the sector for a given ticker.

    Args:
        tickers (str): Stock ticker symbol.

    Returns:
        str or None: Sector name if available, otherwise None.
    """
    try:
        info = yf.Ticker(tickers).info
        industry = info.get('sectorKey')
        return industry
    except Exception as e:
        print(f"Error fetching sector for {tickers}: {e}")
        return None


def benjamin_graham_iv(ticker):
    """
    Calculate intrinsic value using Benjamin Graham's formula.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Calculated intrinsic value, rounded to 2 decimal places.
    """
    np.printoptions(suppress=True)
    info = yf.Ticker(ticker).info
    eps = info.get('epsTrailingTwelveMonths')
    pe_ratios = pe_ratio(ticker)
    g = cagr_eps(ticker)
    intrinsic_value_1 = eps * (12 + 2 * g) * (float(pe_ratios) / 21.59)
    result = round(intrinsic_value_1, 2)
    return result


def dcf_intrinsic_value(tickers):
    """
    Calculate intrinsic value using Discounted Cash Flow (DCF) method.

    Args:
        tickers (str): Stock ticker symbol.

    Returns:
        float: DCF-based intrinsic value per share, rounded to 2 decimal places.
    """
    np.set_printoptions(suppress=True)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    intrinsic_value = []
    try:
        stock = yf.Ticker(tickers)
        info = stock.info
        financial = stock.financials
        sheet = stock.balancesheet
        income_stmt = stock.income_stmt

        shares_outstanding = info.get('sharesOutstanding')
        terminal_growth = 0.03
        market_cap = info.get('marketCap')

        # Total Debt
        if 'Total Debt' in sheet.index:
            total_debt_series = sheet.loc['Total Debt'].dropna()
            total_debt = total_debt_series.iloc[0] if not total_debt_series.empty else 0
        else:
            total_debt = info.get('totalDebt', 0) or 0

        # CAPM calculation
        Rf = 0.04
        beta = info.get("beta") or beta_info(tickers)
        Rm = 0.125
        capm = Rf + beta * (Rm - Rf)

        # Cost of Debt
        interest_expense = income_stmt.loc['Interest Expense'] if 'Interest Expense' in income_stmt.index else None
        if interest_expense is None or interest_expense.empty:
            Rd = 0
        else:
            interest_expense_val = interest_expense.dropna().iloc[0]
            Rd = interest_expense_val / total_debt if total_debt != 0 else 0

        # Tax Rate
        tax_provision = financial.loc['Tax Provision'] if 'Tax Provision' in financial.index else None
        pretax_income = financial.loc['Pretax Income'] if 'Pretax Income' in financial.index else None
        if (tax_provision is None or tax_provision.empty or
            pretax_income is None or pretax_income.empty or pretax_income.dropna().iloc[0] == 0):
            tax_rate = 0.21
        else:
            tax_rate = tax_provision.dropna().iloc[0] / pretax_income.dropna().iloc[0]

        # WACC weights
        total_value = market_cap + total_debt if (market_cap and total_debt) else 1
        weight_equity = market_cap / total_value if total_value != 0 else 0
        weight_debt = total_debt / total_value if total_value != 0 else 0

        wacc = (weight_equity * capm) + (weight_debt * Rd * (1 - tax_rate))

        # Projected Free Cash Flow
        fcf = project_fcf(tickers)
        years = len(fcf)
        pv_fcf = [fcf[t] / (1 + wacc) ** (t + 1) for t in range(years)]

        # Terminal value
        if wacc == terminal_growth:
            tv = 0
        else:
            tv = (fcf[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)

        pv_tv = tv / (1 + wacc) ** years

        dcf_value = sum(pv_fcf) + pv_tv

        # Adjust for cash
        if 'Cash And Cash Equivalents' in sheet.index:
            cash_series = sheet.loc['Cash And Cash Equivalents'].dropna()
            cash = cash_series.iloc[0] if not cash_series.empty else 0
        else:
            cash = 0

        equity_value = dcf_value - total_debt + cash
        if shares_outstanding:
            intrinsic_per_share = equity_value / shares_outstanding
        else:
            intrinsic_per_share = 0

        intrinsic_value.append(intrinsic_per_share)
    except Exception as e:
        print(f"Error calculating DCF for {tickers}: {e}")
        intrinsic_value.append(0)
    return round(intrinsic_value[0], 2)


def fcf_margin_average(ticker):
    """
    Calculate the average Free Cash Flow margin.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Average FCF margin, rounded to 2 decimal places. Returns 0.0 if data is unavailable.
    """
    try:
        fcf = get_fcf(ticker)
        income_stmt = yf.Ticker(ticker).income_stmt
        total_revenue = income_stmt.loc["Total Revenue"].dropna()
        if total_revenue.empty:
            return 0.0
        average_margin = (fcf / total_revenue.iloc[0]).mean()
        return round(average_margin, 2)
    except Exception as e:
        print(f"Error calculating FCF margin average for {ticker}: {e}")
        return 0.0


def operating_margin(ticker):
    """
    Calculate the operating margin (EBIT / Revenue).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: Operating margin rounded to 2 decimals, or None if unavailable.
    """
    try:
        income_stmt = yf.Ticker(ticker).income_stmt
        ebit_series = income_stmt.loc["EBIT"].dropna()
        revenue_series = income_stmt.loc["Total Revenue"].dropna()
        if ebit_series.empty or revenue_series.empty:
            return None
        margin = ebit_series.iloc[0] / revenue_series.iloc[0]
        return round(margin, 2)
    except Exception as e:
        print(f"Error calculating operating margin for {ticker}: {e}")
        return None


def d_ebitda(ticker):
    """
    Calculate the Debt/EBITDA ratio.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: Debt to EBITDA ratio rounded to 2 decimals, or None if unavailable.
    """
    try:
        sheet = yf.Ticker(ticker).balancesheet
        income_stmt = yf.Ticker(ticker).income_stmt
        total_debt_series = sheet.loc['Total Debt'].dropna() if 'Total Debt' in sheet.index else None
        ebitda_series = income_stmt.loc['EBITDA'].dropna() if 'EBITDA' in income_stmt.index else None
        if total_debt_series is None or ebitda_series is None or total_debt_series.empty or ebitda_series.empty:
            return None
        margin = total_debt_series.iloc[0] / ebitda_series.iloc[0]
        return round(margin, 2)
    except Exception as e:
        print(f"Error calculating debt to EBITDA ratio for {ticker}: {e}")
        return None


def fcf_yield(ticker):
    """
    Calculate Free Cash Flow Yield (FCF / Market Cap).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: FCF yield rounded to 2 decimals, or 0.0 if unavailable.
    """
    try:
        fcf = get_fcf(ticker)
        market_cap = yf.Ticker(ticker).info.get('marketCap')
        if market_cap and market_cap != 0:
            fcf_val = fcf.iloc[0] if hasattr(fcf, 'iloc') else fcf
            return round(fcf_val / market_cap, 2)
        return 0.0
    except Exception as e:
        print(f"Error calculating FCF yield for {ticker}: {e}")
        return 0.0


def gross_margin(ticker):
    """
    Calculate the gross margin ((Revenue - COGS) / Revenue).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: Gross margin rounded to 2 decimals, or None if unavailable.
    """
    try:
        income_stmt = yf.Ticker(ticker).income_stmt
        revenue_series = income_stmt.loc["Total Revenue"].dropna()
        cogs_series = income_stmt.loc["Cost Of Revenue"].dropna()
        if revenue_series.empty or cogs_series.empty:
            return None
        margin = (revenue_series.iloc[0] - cogs_series.iloc[0]) / revenue_series.iloc[0]
        return round(margin, 2)
    except Exception as e:
        print(f"Error calculating gross margin for {ticker}: {e}")
        return None


def div_payout_ratio(ticker):
    """
    Calculate the dividend payout ratio (Dividend / EPS).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Dividend payout ratio rounded to 2 decimals, or 0.0 if unavailable.
    """
    try:
        dividends = yf.Ticker(ticker).dividends.dropna()
        div = dividends.iloc[-1] if not dividends.empty else 0
        income_stmt = yf.Ticker(ticker).income_stmt
        eps_series = income_stmt.loc["Basic EPS"].dropna()
        eps = eps_series.iloc[0] if not eps_series.empty else 0
        payout_ratio = div / eps if eps != 0 else 0
        return round(payout_ratio, 2)
    except Exception as e:
        print(f"Error calculating dividend payout ratio for {ticker}: {e}")
        return 0.0


def equity_to_asset(ticker):
    """
    Calculate the Equity-to-Asset ratio (Equity / Total Assets).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: Equity to asset ratio, or None if unavailable.
    """
    try:
        sheet = yf.Ticker(ticker).balancesheet
        shareholders_equity_series = sheet.loc["Stockholders Equity"].dropna() if "Stockholders Equity" in sheet.index else None
        total_assets_series = sheet.loc["Total Assets"].dropna() if "Total Assets" in sheet.index else None
        if shareholders_equity_series is None or total_assets_series is None or shareholders_equity_series.empty or total_assets_series.empty:
            return None
        return shareholders_equity_series.iloc[0] / total_assets_series.iloc[0]
    except Exception as e:
        print(f"Error calculating equity to asset ratio for {ticker}: {e}")
        return None


def roa(ticker):
    """
    Calculate the Return on Assets (ROA).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: ROA rounded to 2 decimals, or 0.0 if unavailable.
    """
    try:
        sheet = yf.Ticker(ticker).balancesheet
        income_stmt = yf.Ticker(ticker).income_stmt
        total_assets_series = sheet.loc["Total Assets"].dropna() if "Total Assets" in sheet.index else None
        if total_assets_series is None or total_assets_series.empty:
            return 0.0
        avg_total_assets = total_assets_series.mean()
        net_income_series = income_stmt.loc["Net Income"].dropna() if "Net Income" in income_stmt.index else None
        if net_income_series is None or net_income_series.empty:
            return 0.0
        net_income = net_income_series.iloc[0]
        calc = net_income / avg_total_assets if avg_total_assets != 0 else 0.0
        return round(calc, 2)
    except Exception as e:
        print(f"Error calculating ROA for {ticker}: {e}")
        return 0.0


def roce(ticker):
    """
    Calculate the Return on Capital Employed (ROCE).

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: ROCE rounded to 2 decimals, or None if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        ebit = stock.income_stmt.loc["EBIT"].iloc[0]
        total_assets = stock.balancesheet.loc["Total Assets"].iloc[0]
        current_liabilities = stock.balancesheet.loc["Current Liabilities"].iloc[0]

        capital_employed = total_assets - current_liabilities
        if capital_employed == 0:
            return None
        return round(ebit / capital_employed, 2)
    except Exception as e:
        print(f"Error calculating ROCE for {ticker}: {e}")
        return None


def ev_ebitda_calc(ticker):
    """
    Calculate the EV/EBITDA ratio.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float or None: EV/EBITDA rounded to 2 decimals, or None if unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get('marketCap', 0)

        total_debt = stock.balancesheet.loc["Total Debt"].dropna()
        cash_equiv = stock.balancesheet.loc["Cash And Cash Equivalents"].dropna()

        total_debt_val = total_debt.iloc[0] if not total_debt.empty else 0
        cash_val = cash_equiv.iloc[0] if not cash_equiv.empty else 0

        enterprise_value = market_cap + total_debt_val - cash_val
        ebitda = stock.income_stmt.loc["EBITDA"].dropna()
        if ebitda.empty or ebitda.iloc[0] == 0:
            return None

        return round(enterprise_value / ebitda.iloc[0], 2)
    except Exception as e:
        print(f"Error calculating EV/EBITDA for {ticker}: {e}")
        return None

def div_coverage_ratio(ticker):
    """
    Calculate the dividend coverage ratio for a given stock ticker.

    Dividend Coverage Ratio = Free Cash Flow / Cash Dividends Paid

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The dividend coverage ratio (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        fcf = get_fcf(ticker)
        if fcf.empty:
            return None
        fcf_val = fcf.iloc[0]

        div_paid = yf.Ticker(ticker).cashflow.loc['Cash Dividends Paid'].dropna()
        if div_paid.empty or div_paid.iloc[0] == 0:
            return None

        div_cover = round(fcf_val / abs(div_paid.iloc[0]), 2)
        return div_cover
    except Exception as e:
        print(f"Error calculating dividend coverage ratio for {ticker}: {e}")
        return None


def inventory_turnover_ratio(ticker):
    """
    Calculate the inventory turnover ratio for a given stock ticker.

    Inventory Turnover = Cost of Revenue / Average Inventory

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The inventory turnover ratio (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        cogs = stock.income_stmt.loc["Cost Of Revenue"].dropna()
        inventory = stock.balancesheet.loc["Inventory"].dropna()

        if cogs.empty or inventory.empty:
            return None

        cogs_val = cogs.iloc[0]
        inventory_mean = inventory.mean()
        if inventory_mean == 0:
            return None

        inv_turnover = round(cogs_val / inventory_mean, 2)
        return inv_turnover
    except Exception as e:
        print(f"Error calculating inventory turnover ratio for {ticker}: {e}")
        return None


def interest_cover_ratio(ticker):
    """
    Calculate the interest coverage ratio for a given stock ticker.

    Interest Coverage Ratio = EBIT / Interest Expense

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The interest coverage ratio (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        ebit = stock.income_stmt.loc["EBIT"].dropna()
        interest_expense = stock.income_stmt.loc['Interest Expense'].dropna()

        if ebit.empty or interest_expense.empty:
            return None

        icr = round((ebit / interest_expense).mean(), 2)
        return icr
    except Exception as e:
        print(f"Error calculating interest coverage ratio for {ticker}: {e}")
        return None


def ffo_cagr(ticker):
    """
    Calculate the growth rate of Funds From Operations (FFO) year-over-year for a given ticker.

    FFO = Net Income + Depreciation & Amortization – Gain on Sale of Securities

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The year-over-year growth of FFO (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        net_income = stock.income_stmt.loc["Net Income"].dropna()
        dep_n_amo = stock.cashflow.loc['Depreciation And Amortization'].dropna()
        gain_on_sales = stock.income_stmt.loc['Gain On Sale Of Security'].dropna()

        if net_income.empty or dep_n_amo.empty or gain_on_sales.empty:
            return None

        ffo = net_income + dep_n_amo - gain_on_sales
        if len(ffo) < 2 or ffo.iloc[1] == 0:
            return None

        ffo_growth = round((ffo.iloc[0] - ffo.iloc[1]) / ffo.iloc[1], 2)
        return ffo_growth
    except Exception as e:
        print(f"Error calculating FFO CAGR for {ticker}: {e}")
        return None


def p_ffo(ticker):
    """
    Calculate the Price to Funds From Operations (P/FFO) ratio for a given ticker.

    P/FFO = Current Stock Price / FFO (most recent period)

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The P/FFO ratio (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        net_income = stock.income_stmt.loc["Net Income"].dropna()
        dep_n_amo = stock.cashflow.loc['Depreciation And Amortization'].dropna()
        gain_on_sales = stock.income_stmt.loc['Gain On Sale Of Security'].dropna()

        if net_income.empty or dep_n_amo.empty or gain_on_sales.empty:
            return None

        ffo = net_income + dep_n_amo - gain_on_sales
        if ffo.empty or ffo.iloc[0] == 0:
            return None

        current_price = current_stock_price(ticker)
        price_to_ffo = round(current_price / ffo.iloc[0], 2)
        return price_to_ffo
    except Exception as e:
        print(f"Error calculating P/FFO for {ticker}: {e}")
        return None


def dividend_pay_ffo(ticker):
    """
    Calculate the dividend payout as a proportion of Funds From Operations (FFO).

    Dividend Payout to FFO = Cash Dividends Paid / FFO (most recent period)

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The dividend payout ratio to FFO (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        net_income = stock.income_stmt.loc["Net Income"].dropna()
        dep_n_amo = stock.cashflow.loc['Depreciation And Amortization'].dropna()
        gain_on_sales = stock.income_stmt.loc['Gain On Sale Of Security'].dropna()
        div_paid = stock.cashflow.loc['Cash Dividends Paid'].dropna()

        if net_income.empty or dep_n_amo.empty or gain_on_sales.empty or div_paid.empty:
            return None

        ffo = net_income + dep_n_amo - gain_on_sales
        if ffo.iloc[0] == 0:
            return None

        payout_ffo = round(div_paid.iloc[0] / ffo.iloc[0], 2)
        return payout_ffo
    except Exception as e:
        print(f"Error calculating dividend payout to FFO for {ticker}: {e}")
        return None


def r_d_ratio(ticker):
    """
    Calculate the Research & Development (R&D) expense ratio for a given stock ticker.

    R&D Ratio = Research and Development Expenses / Total Revenue

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    float or None
        The R&D ratio (rounded to 2 decimals) if available, otherwise None.
    """
    try:
        stock = yf.Ticker(ticker)
        r_development = stock.income_stmt.loc['Research And Development'].dropna()
        revenue = stock.income_stmt.loc['Total Revenue'].dropna()

        if r_development.empty or revenue.empty:
            return None

        ratio = round((r_development / revenue).mean(), 2)
        return ratio
    except Exception as e:
        print(f"Error calculating R&D ratio for {ticker}: {e}")
        return None
