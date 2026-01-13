"""
Training Scheduler - Automated model retraining
Uses APScheduler for periodic retraining jobs
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import asyncio
from dataclasses import dataclass

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

from app.config import get_settings
from app.ml.training.model_store import model_store


@dataclass
class TrainingJob:
    """Training job configuration"""
    symbol: str
    model_name: str
    schedule: str  # Cron expression or 'daily', 'weekly'
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = "pending"


class TrainingScheduler:
    """
    Automated Training Scheduler

    Schedules periodic model retraining based on:
    - Time-based schedules (daily, weekly)
    - Model staleness (age-based)
    - Performance degradation (accuracy-based)
    """

    def __init__(self):
        self.settings = get_settings()
        self.scheduler = None
        self.jobs: Dict[str, TrainingJob] = {}
        self._training_callback: Optional[Callable] = None

        if SCHEDULER_AVAILABLE:
            self.scheduler = AsyncIOScheduler()
        else:
            print("APScheduler not installed - scheduling disabled")

    def set_training_callback(self, callback: Callable):
        """
        Set callback function for training

        Args:
            callback: Async function(symbol, model_name) to train models
        """
        self._training_callback = callback

    def start(self):
        """Start the scheduler"""
        if self.scheduler and not self.scheduler.running:
            self.scheduler.start()
            print("Training scheduler started")

    def stop(self):
        """Stop the scheduler"""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown()
            print("Training scheduler stopped")

    def add_job(
        self,
        symbol: str,
        model_name: str = "ensemble",
        schedule: str = "weekly"
    ) -> str:
        """
        Add a training job

        Args:
            symbol: Stock symbol
            model_name: Model to train
            schedule: 'daily', 'weekly', or cron expression

        Returns:
            Job ID
        """
        job_id = f"{symbol}_{model_name}"

        job = TrainingJob(
            symbol=symbol,
            model_name=model_name,
            schedule=schedule,
        )
        self.jobs[job_id] = job

        if self.scheduler and self._training_callback:
            # Convert schedule to cron trigger
            if schedule == "daily":
                trigger = CronTrigger(hour=1, minute=0)  # 1 AM daily
            elif schedule == "weekly":
                trigger = CronTrigger(day_of_week=0, hour=1, minute=0)  # Monday 1 AM
            else:
                trigger = CronTrigger.from_crontab(schedule)

            self.scheduler.add_job(
                self._run_training,
                trigger=trigger,
                args=[symbol, model_name],
                id=job_id,
                replace_existing=True
            )

            # Calculate next run
            job.next_run = trigger.get_next_fire_time(None, datetime.now())

        return job_id

    def remove_job(self, symbol: str, model_name: str = "ensemble") -> bool:
        """Remove a training job"""
        job_id = f"{symbol}_{model_name}"

        if job_id in self.jobs:
            del self.jobs[job_id]

            if self.scheduler:
                try:
                    self.scheduler.remove_job(job_id)
                except:
                    pass

            return True
        return False

    async def _run_training(self, symbol: str, model_name: str):
        """Execute training job"""
        job_id = f"{symbol}_{model_name}"

        if job_id in self.jobs:
            self.jobs[job_id].status = "running"
            self.jobs[job_id].last_run = datetime.now()

        try:
            if self._training_callback:
                await self._training_callback(symbol, model_name)
                print(f"Training completed for {symbol}/{model_name}")

            if job_id in self.jobs:
                self.jobs[job_id].status = "completed"

        except Exception as e:
            print(f"Training failed for {symbol}/{model_name}: {e}")
            if job_id in self.jobs:
                self.jobs[job_id].status = "failed"

    async def train_stale_models(
        self,
        symbols: List[str],
        max_age_hours: int = 168  # 1 week
    ) -> List[str]:
        """
        Train all stale models

        Args:
            symbols: List of symbols to check
            max_age_hours: Maximum model age

        Returns:
            List of trained symbols
        """
        trained = []

        for symbol in symbols:
            if model_store.is_model_stale(symbol, "ensemble", max_age_hours):
                try:
                    await self._run_training(symbol, "ensemble")
                    trained.append(symbol)
                except Exception as e:
                    print(f"Failed to train {symbol}: {e}")

        return trained

    def get_job_status(self, symbol: str, model_name: str = "ensemble") -> Optional[Dict]:
        """Get status of a training job"""
        job_id = f"{symbol}_{model_name}"
        job = self.jobs.get(job_id)

        if not job:
            return None

        return {
            "symbol": job.symbol,
            "model_name": job.model_name,
            "schedule": job.schedule,
            "status": job.status,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "next_run": job.next_run.isoformat() if job.next_run else None,
        }

    def get_all_jobs(self) -> List[Dict]:
        """Get status of all jobs"""
        return [
            self.get_job_status(job.symbol, job.model_name)
            for job in self.jobs.values()
        ]

    async def trigger_immediate_training(
        self,
        symbol: str,
        model_name: str = "ensemble"
    ) -> bool:
        """Trigger immediate training for a symbol"""
        try:
            await self._run_training(symbol, model_name)
            return True
        except Exception as e:
            print(f"Immediate training failed: {e}")
            return False


# Default training callback
async def default_training_callback(symbol: str, model_name: str):
    """Default training implementation"""
    import yfinance as yf
    from app.ml.prediction import EnsemblePredictor
    from app.ml.features import technical_feature_generator

    print(f"Training {model_name} for {symbol}...")

    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")

    if df.empty:
        raise ValueError(f"No data for {symbol}")

    # Generate features
    features = technical_feature_generator.generate_features(df)
    target = df['Close'].pct_change(20).shift(-20)
    prices = df['Close']

    # Align
    valid_idx = features.index.intersection(target.dropna().index)
    features = features.loc[valid_idx].ffill().bfill()
    target = target.loc[valid_idx]
    prices = prices.loc[valid_idx]

    # Train
    ensemble = EnsemblePredictor()
    metrics = ensemble.train_all(features, target, prices)

    # Save
    aggregated_metrics = {}
    for name, m in metrics.items():
        if m.directional_accuracy:
            aggregated_metrics[f"{name}_accuracy"] = m.directional_accuracy

    model_store.save_model(
        model=ensemble,
        symbol=symbol,
        model_name=model_name,
        metrics=aggregated_metrics,
        feature_count=len(features.columns),
        train_samples=len(features)
    )

    print(f"Training complete: {aggregated_metrics}")


# Global instance
training_scheduler = TrainingScheduler()
training_scheduler.set_training_callback(default_training_callback)
