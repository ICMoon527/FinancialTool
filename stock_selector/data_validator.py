# -*- coding: utf-8 -*-
"""
Stock Data Validation and Preparation Module.

Provides functionality to:
1. Validate stock data completeness for a given time range
2. Automatically fetch missing/incomplete stock data
3. Retry failed downloads with exponential backoff
4. Log detailed data preparation process
"""

import logging
from datetime import date, timedelta, datetime
from typing import List, Tuple, Set, Dict, Optional
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.storage import DatabaseManager, StockDaily
from sqlalchemy import select, and_, func, or_
from data_provider import DataFetcherManager, DataFetchError

logger = logging.getLogger(__name__)


class StockDataValidator:
    """
    Validates and prepares stock data for Walk-Forward Analysis.
    """

    # Required fields for completeness
    REQUIRED_FIELDS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        data_fetcher_manager: Optional[DataFetcherManager] = None
    ):
        """
        Initialize the validator.

        Args:
            db_manager: Database manager instance
            data_fetcher_manager: Data fetcher manager instance
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.data_fetcher_manager = data_fetcher_manager or DataFetcherManager()
        self.stock_repo = self._init_stock_repo()

    def _init_stock_repo(self):
        """Initialize stock repository."""
        from src.repositories.stock_repo import StockRepository
        return StockRepository(self.db_manager)

    def get_stocks_with_data(self) -> Set[str]:
        """
        Get all stock codes that have at least some data in database.

        Returns:
            Set of stock codes with data
        """
        try:
            with self.db_manager.get_session() as session:
                stmt = select(func.distinct(StockDaily.code))
                codes = set(session.execute(stmt).scalars().all())
                logger.info(f"Found {len(codes)} stocks with existing data")
                return codes
        except Exception as e:
            logger.error(f"Failed to get stocks with data: {e}")
            return set()

    def validate_stock_data(
        self,
        stock_code: str,
        start_date: date,
        end_date: date
    ) -> Tuple[bool, List[date]]:
        """
        Validate if a stock has complete data for the given date range.

        Args:
            stock_code: Stock code to validate
            start_date: Start of validation period
            end_date: End of validation period

        Returns:
            Tuple of (is_complete, list_of_missing_dates)
        """
        try:
            data = self.stock_repo.get_range(stock_code, start_date, end_date)

            if not data:
                logger.debug(f"Stock {stock_code} has no data for {start_date} to {end_date}")
                missing_dates = self._get_all_dates_in_range(start_date, end_date)
                return False, missing_dates

            # Check for completeness of required fields
            complete_dates = set()
            for item in data:
                if all(getattr(item, field) is not None for field in self.REQUIRED_FIELDS):
                    complete_dates.add(item.date)

            all_dates = self._get_all_dates_in_range(start_date, end_date)
            missing_dates = [d for d in all_dates if d not in complete_dates]

            is_complete = len(missing_dates) == 0

            if not is_complete:
                logger.debug(f"Stock {stock_code} missing {len(missing_dates)} dates")
            else:
                logger.debug(f"Stock {stock_code} has complete data for {start_date} to {end_date}")

            return is_complete, missing_dates

        except Exception as e:
            logger.error(f"Error validating stock {stock_code}: {e}")
            return False, self._get_all_dates_in_range(start_date, end_date)

    def _get_all_dates_in_range(self, start_date: date, end_date: date) -> List[date]:
        """
        Get all dates in a given range (calendar days, not just trading days).

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of dates
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((DataFetchError, Exception))
    )
    def fetch_and_save_stock_data(
        self,
        stock_code: str,
        start_date: date,
        end_date: date
    ) -> bool:
        """
        Fetch stock data from data sources and save to database.

        Args:
            stock_code: Stock code
            start_date: Start date
            end_date: End date

        Returns:
            True if successful
        """
        try:
            logger.info(f"Fetching data for {stock_code}: {start_date} to {end_date}")

            df, source = self.data_fetcher_manager.get_daily_data(
                stock_code=stock_code,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if df is None or df.empty:
                logger.warning(f"No data fetched for {stock_code}")
                return False

            # Save to database
            saved_count = self._save_dataframe_to_database(df, stock_code)
            logger.info(f"Saved {saved_count} records for {stock_code} from {source}")

            return saved_count > 0

        except Exception as e:
            logger.error(f"Failed to fetch/save data for {stock_code}: {e}")
            raise

    def _save_dataframe_to_database(self, df, stock_code: str) -> int:
        """
        Save DataFrame data to database.

        Args:
            df: DataFrame with stock data
            stock_code: Stock code

        Returns:
            Number of records saved
        """
        saved_count = 0
        try:
            with self.db_manager.get_session() as session:
                for _, row in df.iterrows():
                    # Get or create StockDaily record
                    record_date = row.get('date')
                    if isinstance(record_date, datetime):
                        record_date = record_date.date()
                    elif isinstance(record_date, str):
                        record_date = datetime.strptime(record_date, '%Y-%m-%d').date()

                    # Check if record already exists
                    existing = session.execute(
                        select(StockDaily)
                        .where(and_(
                            StockDaily.code == stock_code,
                            StockDaily.date == record_date
                        ))
                    ).scalar_one_or_none()

                    if existing:
                        # Update existing record
                        existing.open = row.get('open')
                        existing.high = row.get('high')
                        existing.low = row.get('low')
                        existing.close = row.get('close')
                        existing.volume = row.get('volume')
                        existing.amount = row.get('amount')
                        existing.pct_chg = row.get('pct_chg')
                    else:
                        # Create new record
                        new_record = StockDaily(
                            code=stock_code,
                            date=record_date,
                            open=row.get('open'),
                            high=row.get('high'),
                            low=row.get('low'),
                            close=row.get('close'),
                            volume=row.get('volume'),
                            amount=row.get('amount'),
                            pct_chg=row.get('pct_chg')
                        )
                        session.add(new_record)
                    saved_count += 1

                session.commit()
                logger.debug(f"Committed {saved_count} records for {stock_code}")
                return saved_count

        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            if 'session' in locals():
                session.rollback()
            return 0

    def prepare_data_for_wfa(
        self,
        stock_codes: List[str],
        wfa_start_date: date,
        training_window_days: int = 60,
        testing_window_days: int = 20,
        num_windows: int = 5,
        step_days: int = 30
    ) -> Dict[str, any]:
        """
        Prepare all stock data for Walk-Forward Analysis.

        Args:
            stock_codes: List of stock codes to prepare
            wfa_start_date: Start date of WFA
            training_window_days: Training window size in days
            testing_window_days: Testing window size in days
            num_windows: Number of WFA windows
            step_days: Step days between windows

        Returns:
            Dictionary with preparation summary
        """
        logger.info("=" * 80)
        logger.info("Starting stock data preparation for WFA")
        logger.info("=" * 80)

        # Calculate the full date range we need data for
        # First window training starts at wfa_start_date
        # Last window testing ends at wfa_start_date + training + (num_windows-1)*step + testing
        # Only need data from (wfa_start_date - training_window_days) to end of last window
        # Add a small buffer (30 days) for safety
        data_start_date = wfa_start_date - timedelta(days=training_window_days + 30)
        
        # Get latest date in database to know end date
        latest_date = self._get_latest_date_in_database()
        if latest_date is None:
            # If no data, use a reasonable end date
            latest_date = date.today()

        logger.info(f"Data preparation range: {data_start_date} to {latest_date}")
        logger.info(f"Total stocks to check: {len(stock_codes)}")

        # Track statistics
        stats = {
            'total_stocks': len(stock_codes),
            'stocks_with_complete_data': 0,
            'stocks_needing_data': 0,
            'stocks_successfully_fetched': 0,
            'stocks_failed_to_fetch': 0,
            'failed_stocks': []
        }

        # Process each stock
        for i, stock_code in enumerate(stock_codes, 1):
            logger.info(f"Processing stock {i}/{len(stock_codes)}: {stock_code}")

            # First check if stock has any data at all
            is_complete, missing_dates = self.validate_stock_data(
                stock_code, data_start_date, latest_date
            )

            if is_complete:
                stats['stocks_with_complete_data'] += 1
                logger.debug(f"Stock {stock_code} already has complete data")
                continue

            # Need to fetch data
            stats['stocks_needing_data'] += 1
            logger.info(f"Stock {stock_code} needs data, fetching...")

            try:
                success = self.fetch_and_save_stock_data(
                    stock_code, data_start_date, latest_date
                )

                if success:
                    stats['stocks_successfully_fetched'] += 1
                    logger.info(f"Successfully fetched data for {stock_code}")
                else:
                    stats['stocks_failed_to_fetch'] += 1
                    stats['failed_stocks'].append(stock_code)
                    logger.warning(f"Failed to fetch data for {stock_code}")

            except Exception as e:
                stats['stocks_failed_to_fetch'] += 1
                stats['failed_stocks'].append(stock_code)
                logger.error(f"Error fetching data for {stock_code}: {e}")

        # Print summary
        logger.info("=" * 80)
        logger.info("Data Preparation Summary")
        logger.info("=" * 80)
        logger.info(f"Total stocks checked: {stats['total_stocks']}")
        logger.info(f"Stocks with complete data: {stats['stocks_with_complete_data']}")
        logger.info(f"Stocks needing data: {stats['stocks_needing_data']}")
        logger.info(f"  - Successfully fetched: {stats['stocks_successfully_fetched']}")
        logger.info(f"  - Failed to fetch: {stats['stocks_failed_to_fetch']}")
        if stats['failed_stocks']:
            logger.warning(f"Failed stocks: {', '.join(stats['failed_stocks'][:20])}")
            if len(stats['failed_stocks']) > 20:
                logger.warning(f"  ... and {len(stats['failed_stocks']) - 20} more")

        logger.info("=" * 80)

        return stats

    def filter_stocks_with_complete_data(
        self,
        stock_codes: List[str],
        start_date: date,
        end_date: date,
        min_data_ratio: float = 0.8
    ) -> Tuple[List[str], Dict[str, any]]:
        """
        Filter stocks that have sufficient data for the given date range.

        Args:
            stock_codes: List of stock codes to filter
            start_date: Start of the date range
            end_date: End of the date range
            min_data_ratio: Minimum ratio of trading days required (0.0-1.0)

        Returns:
            Tuple of (filtered_stock_codes, statistics_dict)
        """
        logger.info("=" * 80)
        logger.info(f"Filtering stocks with complete data from {start_date} to {end_date}")
        logger.info("=" * 80)

        stats = {
            'total_stocks': len(stock_codes),
            'stocks_passed': 0,
            'stocks_failed': 0,
            'failed_stocks': [],
            'date_range_start': start_date,
            'date_range_end': end_date,
            'min_data_ratio': min_data_ratio
        }

        filtered_stocks = []

        for i, stock_code in enumerate(stock_codes, 1):
            logger.debug(f"Checking stock {i}/{len(stock_codes)}: {stock_code}")

            try:
                data = self.stock_repo.get_range(stock_code, start_date, end_date)

                if not data:
                    stats['stocks_failed'] += 1
                    stats['failed_stocks'].append(stock_code)
                    logger.debug(f"Stock {stock_code}: no data in range")
                    continue

                # Calculate data completeness ratio
                # We check required fields completeness
                complete_records = 0
                for item in data:
                    if all(getattr(item, field) is not None for field in self.REQUIRED_FIELDS):
                        complete_records += 1

                # Estimate total trading days (approximate)
                total_days_needed = (end_date - start_date).days
                # Assume ~250 trading days per year
                estimated_trading_days = int(total_days_needed * 5 / 7)  # ~5 trading days per week
                if estimated_trading_days < 1:
                    estimated_trading_days = 1

                data_ratio = complete_records / estimated_trading_days

                if data_ratio >= min_data_ratio:
                    filtered_stocks.append(stock_code)
                    stats['stocks_passed'] += 1
                    logger.debug(f"Stock {stock_code}: passed (ratio={data_ratio:.2f})")
                else:
                    stats['stocks_failed'] += 1
                    stats['failed_stocks'].append(stock_code)
                    logger.debug(f"Stock {stock_code}: failed (ratio={data_ratio:.2f} < {min_data_ratio})")

            except Exception as e:
                stats['stocks_failed'] += 1
                stats['failed_stocks'].append(stock_code)
                logger.error(f"Error checking stock {stock_code}: {e}")

        # Log summary
        logger.info("=" * 80)
        logger.info("Stock Filtering Summary")
        logger.info("=" * 80)
        logger.info(f"Total stocks checked: {stats['total_stocks']}")
        logger.info(f"Stocks with sufficient data: {stats['stocks_passed']}")
        logger.info(f"Stocks with insufficient data: {stats['stocks_failed']}")
        logger.info(f"Pass rate: {stats['stocks_passed']/stats['total_stocks']*100:.1f}%")
        if stats['failed_stocks']:
            logger.warning(f"Failed stocks (first 20): {', '.join(stats['failed_stocks'][:20])}")
            if len(stats['failed_stocks']) > 20:
                logger.warning(f"  ... and {len(stats['failed_stocks']) - 20} more")
        logger.info("=" * 80)

        return filtered_stocks, stats

    def calculate_wfa_date_range(
        self,
        wfa_start_date: date,
        training_window_days: int = 60,
        testing_window_days: int = 20,
        num_windows: int = 5,
        step_days: int = 10
    ) -> Tuple[date, date]:
        """
        Calculate the full date range needed for WFA.

        Args:
            wfa_start_date: Start date of WFA
            training_window_days: Training window size
            testing_window_days: Testing window size
            num_windows: Number of windows
            step_days: Step between windows

        Returns:
            Tuple of (full_start_date, full_end_date)
        """
        # First window: training starts at wfa_start_date
        # Last window: training starts at wfa_start_date + (num_windows-1)*step_days
        last_training_start = wfa_start_date + timedelta(days=(num_windows - 1) * step_days)
        last_training_end = last_training_start + timedelta(days=training_window_days - 1)
        last_testing_end = last_training_end + timedelta(days=testing_window_days)

        # Only need data from (wfa_start_date - training_window_days) to end of last window
        # Add a small buffer (30 days) for safety
        full_start = wfa_start_date - timedelta(days=training_window_days + 30)
        full_end = last_testing_end

        logger.info(f"WFA full date range: {full_start} to {full_end}")

        return full_start, full_end

    def _get_latest_date_in_database(self) -> Optional[date]:
        """
        Get the latest date present in the database.

        Returns:
            Latest date or None
        """
        try:
            with self.db_manager.get_session() as session:
                stmt = select(func.max(StockDaily.date))
                return session.execute(stmt).scalar()
        except Exception as e:
            logger.error(f"Failed to get latest date: {e}")
            return None


def prepare_data_for_wfa(
    stock_codes: List[str],
    wfa_start_date: date,
    training_window_days: int = 60,
    testing_window_days: int = 20,
    num_windows: int = 5,
    step_days: int = 30
) -> Dict[str, any]:
    """
    Convenience function to prepare data for WFA.

    Args:
        stock_codes: List of stock codes
        wfa_start_date: Start date of WFA
        training_window_days: Training window days
        testing_window_days: Testing window days
        num_windows: Number of windows
        step_days: Step days between windows

    Returns:
        Preparation summary dictionary
    """
    validator = StockDataValidator()
    return validator.prepare_data_for_wfa(
        stock_codes=stock_codes,
        wfa_start_date=wfa_start_date,
        training_window_days=training_window_days,
        testing_window_days=testing_window_days,
        num_windows=num_windows,
        step_days=step_days
    )


def filter_stocks_for_wfa(
    stock_codes: List[str],
    wfa_start_date: date,
    training_window_days: int = 60,
    testing_window_days: int = 20,
    num_windows: int = 5,
    step_days: int = 10,
    min_data_ratio: float = 0.8
) -> Tuple[List[str], Dict[str, any]]:
    """
    Convenience function to filter stocks for WFA.

    Args:
        stock_codes: List of stock codes
        wfa_start_date: Start date of WFA
        training_window_days: Training window days
        testing_window_days: Testing window days
        num_windows: Number of windows
        step_days: Step between windows
        min_data_ratio: Minimum data ratio required

    Returns:
        Tuple of (filtered_stock_codes, statistics_dict)
    """
    validator = StockDataValidator()
    full_start, full_end = validator.calculate_wfa_date_range(
        wfa_start_date=wfa_start_date,
        training_window_days=training_window_days,
        testing_window_days=testing_window_days,
        num_windows=num_windows,
        step_days=step_days
    )
    return validator.filter_stocks_with_complete_data(
        stock_codes=stock_codes,
        start_date=full_start,
        end_date=full_end,
        min_data_ratio=min_data_ratio
    )