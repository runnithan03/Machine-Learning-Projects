-- Create a table to store financial data
CREATE TABLE financial_data (
    date DATE,
    ticker VARCHAR(10),
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    return DECIMAL(10, 4),
    risk DECIMAL(10, 4)
);

-- Insert sample data (you would typically import this from a CSV file)
INSERT INTO financial_data (date, ticker, open, high, low, close, volume, return, risk)
VALUES 
    ('2023-01-01', 'AAPL', 150.00, 152.00, 149.00, 151.00, 1000000, 0.0067, 0.0150),
    ('2023-01-02', 'AAPL', 151.00, 153.00, 150.00, 152.00, 1100000, 0.0066, 0.0148),
    ('2023-01-01', 'GOOGL', 2800.00, 2820.00, 2790.00, 2810.00, 500000, 0.0036, 0.0180),
    ('2023-01-02', 'GOOGL', 2810.00, 2830.00, 2800.00, 2820.00, 550000, 0.0036, 0.0175);

-- Calculate average daily return and risk for each stock
SELECT 
    ticker,
    AVG(return) AS  avg_daily_return,
    STDDEV(return) AS daily_risk
FROM 
    financial_data
GROUP BY 
    ticker;

-- Calculate correlation between stocks
WITH daily_returns AS (
    SELECT 
         
        date,
        ticker,
        return
    FROM 
        financial_data
)
SELECT 
    a.ticker AS stock_a,
    b.ticker AS stock_b,
    CORR(a.return, b.return) AS correlation
FROM 
    daily_returns a
JOIN 
    daily_returns b ON a.date = b.date AND a.ticker < b.ticker
GROUP BY 
    a.ticker, b.ticker;

-- Calculate Sharpe ratio (assuming risk-free rate of 0.02)
SELECT 
    ticker,
    (AVG(return) - 0.02) / STDDEV(return) AS sharpe_ratio
FROM 
    financial_data
GROUP BY 
    ticker;

-- Identify top performing stocks based on Sharpe ratio
WITH stock_performance AS (
    SELECT 
        ticker,
        (AVG(return) - 0.02) / STDDEV(return) AS sharpe_ratio
    FROM 
        financial_data
    GROUP BY 
        ticker
)
SELECT 
    ticker,
    sharpe_ratio
FROM 
    stock_performance
ORDER BY 
    sharpe_ratio DESC
LIMIT 5;
