import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib.font_manager as fm
import os

# 한국어 폰트 설정 (Windows: Malgun Gothic)
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font = fm.FontProperties(fname=font_path)
    plt.rc('font', family='Malgun Gothic')
else:
    logging.warning("한국어 폰트(Malgun Gothic)를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# 로깅 설정 (콘솔 출력 제거, 파일에만 기록)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log')
    ]
)

# 데이터 캐싱을 위한 딕셔너리
data_cache = {}

# MDD 계산
def calculate_mdd(prices):
    if not prices or not isinstance(prices, (list, np.ndarray)):
        logging.warning("MDD 계산: 입력 데이터가 비어 있거나 유효하지 않음")
        return 0
    prices = np.array(prices)
    if np.any(prices <= 0):
        logging.warning("MDD 계산: 가격 데이터에 0 이하 값 존재")
        return 0
    peak = np.maximum.accumulate(prices)
    drawdowns = (peak - prices) / peak * 100
    return max(drawdowns) if len(drawdowns) > 0 else 0

# 주가 데이터 가져오기 (캐싱 적용)
def get_stock_data(symbol, start_date, end_date):
    cache_key = (symbol, start_date, end_date)
    if cache_key in data_cache:
        logging.info(f"{symbol} 데이터 캐시에서 로드")
        return data_cache[cache_key]

    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start, end=end)
        if hist.empty:
            logging.warning(f"{symbol}의 주가 데이터가 비어 있음")
            return []
        data = hist['Close'].tolist()
        data_cache[cache_key] = data
        logging.info(f"{symbol}의 주가 데이터 캐싱 완료")
        return data
    except Exception as e:
        logging.error(f"{symbol} 데이터 가져오기 실패: {e}")
        return []

# 실시간 주가 및 최근 데이터 가져오기 (캐싱 적용)
def get_current_stock_data(symbol, period='1y'):
    cache_key = (symbol, 'current_data')
    if cache_key in data_cache:
        logging.info(f"{symbol} 최근 데이터 캐시에서 로드")
        return data_cache[cache_key]

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            logging.warning(f"{symbol}의 최근 데이터가 비어 있음")
            return None
        data_cache[cache_key] = hist
        logging.info(f"{symbol}의 최근 데이터 캐싱 완료")
        return hist
    except Exception as e:
        logging.error(f"{symbol} 최근 데이터 가져오기 실패: {e}")
        return None

# RSI 계산
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 연간 수익률 계산
def calculate_annual_return(prices):
    if len(prices) < 2 or any(p <= 0 for p in prices):
        logging.warning("연간 수익률 계산: 데이터 부족 또는 유효하지 않은 가격")
        return 0
    return (prices[-1] - prices[0]) / prices[0] * 100

# 연평균 주간 변동성 계산 (캐싱 적용)
def calculate_annual_weekly_volatility(ticker):
    cache_key = (ticker, 'volatility')
    if cache_key in data_cache:
        logging.info(f"{ticker} 변동성 캐시에서 로드")
        return data_cache[cache_key]

    try:
        data = yf.download(ticker, period='max', auto_adjust=False)
        if data.empty or len(data) < 2:
            logging.warning(f"{ticker} 변동성 계산: 데이터 부족")
            return 0
        data.index = pd.to_datetime(data.index)
        price_column = 'Close'
        weekly_returns = data[price_column].resample('W').ffill().pct_change().dropna()
        if len(weekly_returns) < 2:
            logging.warning(f"{ticker} 변동성 계산: 주간 데이터 부족")
            return 0
        weekly_volatility = np.std(weekly_returns.to_numpy(), ddof=1)
        result = weekly_volatility * np.sqrt(52) * 100
        data_cache[cache_key] = result
        logging.info(f"{ticker}의 변동성 캐싱 완료: {result:.2f}%")
        return result
    except Exception as e:
        logging.error(f"{ticker} 변동성 계산 실패: {e}")
        return 0

# 시장 트렌드 판단 (20일 SMA vs 50일 SMA)
def determine_market_trend(symbol):
    hist = get_current_stock_data(symbol, period='1y')
    if hist is None:
        logging.warning(f"{symbol} 시장 트렌드 판단 실패: 데이터 부족")
        return 'neutral'

    close_prices = hist['Close']
    sma20 = close_prices.rolling(window=20).mean().iloc[-1]
    sma50 = close_prices.rolling(window=50).mean().iloc[-1]

    if sma20 > sma50:
        logging.info(f"{symbol} 상승장: 20일 SMA({sma20:.2f}) > 50일 SMA({sma50:.2f})")
        return 'bullish'
    else:
        logging.info(f"{symbol} 하락장: 20일 SMA({sma20:.2f}) <= 50일 SMA({sma50:.2f})")
        return 'bearish'

# 종목 유효성 검증
def validate_symbol(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info or 'symbol' not in info:
            logging.warning(f"{symbol}은 유효하지 않은 종목 코드")
            return False
        return True
    except Exception as e:
        logging.error(f"{symbol} 유효성 검증 실패: {e}")
        return False

# 시각화: MDD, 수익률, 변동성 비교
def plot_stock_metrics(valid_symbols, mdd_dict, annual_return_dict, volatility_dict):
    metrics = ['MDD', 'Annual Return', 'Volatility']
    x = np.arange(len(valid_symbols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, [mdd_dict.get(s, 0) for s in valid_symbols], width, label='MDD (%)')
    ax.bar(x, [annual_return_dict.get(s, 0) for s in valid_symbols], width, label='Annual Return (%)')
    ax.bar(x + width, [volatility_dict.get(s, 0) for s in valid_symbols], width, label='Volatility (%)')

    ax.set_xlabel('종목')
    ax.set_ylabel('값 (%)')
    ax.set_title('종목 지표 비교')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_symbols)
    ax.legend()

    plt.tight_layout()
    plt.savefig('stock_metrics.png')
    plt.close()
    logging.info("종목 지표 그래프가 'stock_metrics.png'로 저장됨")

# 시각화: 매수 전략
def plot_buy_strategy(symbol, mdd_values, investment_amounts):
    stages = range(1, 11)
    mdd_diffs = [mdd_values[0]] + [mdd_values[i] - mdd_values[i-1] for i in range(1, len(mdd_values))]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('차수')
    ax1.set_ylabel('투자 금액 ($)', color='tab:blue')
    ax1.plot(stages, investment_amounts, 'o-', color='tab:blue', label='투자 금액 ($)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('하락 (%)', color='tab:red')
    ax2.plot(stages, mdd_diffs, 's--', color='tab:red', label='하락 (%)')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle(f'{symbol} 매수 전략')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.savefig(f'buy_strategy_{symbol}.png')
    plt.close()
    logging.info(f"{symbol} 매수 전략 그래프가 'buy_strategy_{symbol}.png'로 저장됨")

# 시각화: 매도 전략
def plot_sell_strategy(symbol, sell_thresholds):
    stages = range(1, 11)

    plt.figure(figsize=(10, 6))
    plt.plot(stages, sell_thresholds, 'o-', color='tab:green', label='매도 기준점 (%)')
    plt.xlabel('차수')
    plt.ylabel('매도 기준점 (%)')
    plt.title(f'{symbol} 매도 전략')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'sell_strategy_{symbol}.png')
    plt.close()
    logging.info(f"{symbol} 매도 전략 그래프가 'sell_strategy_{symbol}.png'로 저장됨")

# 종목 리스트 및 날짜 설정
stock_symbols = ['RKLB', 'IONQ']  # 2종목
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

# 유효한 종목만 필터링
valid_symbols = [symbol for symbol in stock_symbols if validate_symbol(symbol)]
if not valid_symbols:
    logging.error("유효한 종목이 없어 프로그램 종료")
    raise ValueError("유효한 종목이 없습니다.")

# 지표 계산
mdd_dict, annual_return_dict, volatility_dict = {}, {}, {}
for symbol in valid_symbols:
    prices = get_stock_data(symbol, start_date, end_date)
    volatility = calculate_annual_weekly_volatility(symbol)
    if prices and volatility > 0:
        mdd_dict[symbol] = calculate_mdd(prices)
        annual_return_dict[symbol] = calculate_annual_return(prices)
        volatility_dict[symbol] = volatility
        logging.info(f"{symbol} 계산 완료 - MDD: {mdd_dict[symbol]:.2f}%, 수익률: {annual_return_dict[symbol]:.2f}%, 변동성: {volatility_dict[symbol]:.2f}%")
    else:
        logging.warning(f"{symbol}의 데이터가 부족하거나 변동성 계산 실패")

# 지표 시각화
plot_stock_metrics(valid_symbols, mdd_dict, annual_return_dict, volatility_dict)

# 투자 설정
total_investment = 20000
num_stocks = len(valid_symbols)
num_stages = 10
investment_amounts = [1000] * 10  # 합계 10000

# 투자 금액 합계 확인
total_per_stock = sum(investment_amounts)
logging.info(f"investment_amounts: {investment_amounts}")
logging.info(f"종목당 투자 금액 합계: {total_per_stock}")

# 종목당 투자 금액 검증 (소수점 차이 허용)
if abs(total_per_stock - total_investment / num_stocks) > 1e-6:
    logging.error(f"종목당 투자 금액 불일치 - 예상: {total_investment / num_stocks}, 실제: {total_per_stock}")
    raise ValueError("종목당 투자 금액이 예상과 일치하지 않습니다.")

# 표 데이터 저장
all_table_data = {}
total_returns = []
stop_loss = -20.0  # 손절 기준

for symbol in valid_symbols:
    trend = determine_market_trend(symbol)
    hist = get_current_stock_data(symbol)
    if hist is None:
        logging.warning(f"{symbol}의 실시간 가격 가져오기 실패")
        continue
    stock_price = hist['Close'].iloc[-1]
    volatility = volatility_dict.get(symbol, 50.0)  # 변동성 없으면 기본값 50%

    # 변동성 기반 퍼센트 설정
    buy_factor = volatility / 10
    sell_factor = volatility / 8
    base_buy_percentages = [buy_factor * x for x in [0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0]]
    base_sell_percentages = [sell_factor * x for x in [0.6, 0.6, 0.6, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2, 1.2]]

    # 시장 트렌드에 따라 조정
    if trend == 'bullish':
        buy_percentages = [p * 0.8 for p in base_buy_percentages]
        sell_percentages = [p * 1.2 for p in base_sell_percentages]
    elif trend == 'bearish':
        buy_percentages = [p * 1.2 for p in base_buy_percentages]
        sell_percentages = [p * 0.8 for p in base_sell_percentages]
    else:
        buy_percentages = base_buy_percentages
        sell_percentages = base_sell_percentages

    # RSI 조정
    rsi = calculate_rsi(hist['Close']).iloc[-1] if 'Close' in hist else 50
    if rsi < 30:
        buy_percentages = [p * 0.9 for p in buy_percentages]
    if rsi > 70:
        sell_percentages = [p * 1.1 for p in sell_percentages]

    # 시가총액 기반 조정
    stock_info = yf.Ticker(symbol).info
    market_cap = stock_info.get('marketCap', 1e9) / 1e9  # 억 달러 단위
    if market_cap < 1:
        buy_percentages = [p * 1.2 for p in buy_percentages]
        sell_percentages = [p * 1.2 for p in sell_percentages]
    elif market_cap < 5:
        buy_percentages = [p * 1.0 for p in buy_percentages]
        sell_percentages = [p * 1.0 for p in sell_percentages]

    # 누적 하락 퍼센트 계산
    mdd_values = []
    cumulative_drop = 0
    for percentage in buy_percentages:
        cumulative_drop += percentage
        mdd_values.append(cumulative_drop)

    # 표 데이터 생성
    table_data = []
    for i in range(1, 11):
        stock_quantity = int(investment_amounts[i-1] / stock_price)
        buy_drop = buy_percentages[i-1] if i == 1 else mdd_values[i-1] - mdd_values[i-2]
        sell_threshold = sell_percentages[i-1]
        buy_price = stock_price * (1 - buy_drop / 100)
        sell_price = stock_price * (1 + sell_threshold / 100)
        # 손절 확인
        if stock_price < buy_price * (1 + stop_loss / 100):
            logging.info(f"{symbol} {i}차수 손절: 주가 {stock_price:.2f} < 손절가 {buy_price * (1 + stop_loss / 100):.2f}")
        table_data.append([
            i,
            trend,
            f"{buy_drop:.1f}%",
            f"${buy_price:.2f}",
            f"{sell_threshold:.1f}%",
            f"${sell_price:.2f}",
            f"${int(investment_amounts[i-1]):,}",
            f"{stock_quantity}주"
        ])
        logging.info(f"{symbol} {i}차수 - 매수 하락: {buy_drop:.1f}%, 매도 상승: {sell_threshold:.1f}%, 금액: {int(investment_amounts[i-1])}, 수량: {stock_quantity}주")

    all_table_data[symbol] = table_data
    total_returns.append((symbol, annual_return_dict.get(symbol, 0)))

    # 시각화
    plot_buy_strategy(symbol, mdd_values, investment_amounts)
    plot_sell_strategy(symbol, sell_percentages)

# 표 출력
headers = ["차수", "시장 트렌드", "매수 하락 %", "주가", "매도 상승 %", "주가", "투자 금액 ($)", "주식 수량"]
for symbol in valid_symbols:
    if symbol in all_table_data:
        print(f"\n{symbol} 매수/매도 전략 표:")
        print(tabulate(all_table_data[symbol], headers=headers, tablefmt="simple"))

# 총 투자 금액 출력
print(f"\n총 투자 금액: ${int(sum(investment_amounts) * num_stocks):,}")

# 수익률 출력
print("\n수익률:")
for symbol, return_rate in total_returns:
    print(f"{symbol} 1년 수익률: {return_rate:.2f}%")
total_portfolio_return = sum(return_rate for _, return_rate in total_returns) / len(total_returns)
print(f"총 포트폴리오 수익률: {total_portfolio_return:.2f}%")