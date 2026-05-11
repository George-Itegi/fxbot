"""
Warmup Trainer
===============
Offline batch training from historical tick data.
Call this before going live to give the model a starting point.

Usage:
  python -m models.warmup_trainer --symbol R_100 --samples 5000
"""

import argparse
import asyncio
import random
from collections import deque

from config import DEFAULT_SYMBOL, OVER_BARRIER, UNDER_BARRIER, CONTRACT_DURATION, get_symbol_decimals
from data.deriv_ws import DerivWS
from data.tick_aggregator import TickAggregator
from data.feature_engine import FeatureEngine
from models.online_learner import OverUnderModel
from utils.logger import setup_logger

logger = setup_logger("models.warmup_trainer")


async def collect_training_data(symbol: str, num_ticks: int = 5000) -> tuple:
    """
    Collect labeled training data from historical ticks.
    
    For each tick where we have enough history:
    - Compute features from current + past ticks
    - Look ahead N ticks to determine label:
      - 1 = at least one tick in next N had digit > barrier
      - 0 = no tick in next N had digit > barrier
    
    Returns:
        (features_list, labels_list) for training
    """
    logger.info(f"Connecting to collect {num_ticks} ticks for {symbol}...")
    
    ws = DerivWS()
    connected = await ws.connect()
    if not connected:
        logger.error("Failed to connect")
        return [], []
    
    # Fetch historical ticks
    history = await ws.get_tick_history(symbol, count=num_ticks + 500)
    await ws.disconnect()
    
    if not history or len(history) < 500:
        logger.error("Not enough historical data")
        return [], []
    
    logger.info(f"Got {len(history)} historical ticks, labeling...")
    
    # Build aggregator and engine
    dp = get_symbol_decimals(symbol)
    agg = TickAggregator(symbol, decimal_places=dp)
    engine = FeatureEngine(agg)
    
    features_list = []
    labels_list = []
    
    # Process ticks, computing features and looking ahead for labels
    all_ticks = history
    look_ahead = CONTRACT_DURATION
    
    for i in range(len(all_ticks)):
        tick = all_ticks[i]
        agg.add_tick(tick["epoch"], tick["quote"], decimal_places=dp)
        
        # Need enough history for features
        if not agg.is_warm("short"):
            continue
        
        # Need enough future ticks for labeling
        if i + look_ahead >= len(all_ticks):
            continue
        
        # Compute features at current tick
        features = engine.compute_features()
        if features is None:
            continue
        
        # Determine label based on Deriv Digit Over/Under rules:
        # The contract resolves on the LAST tick of the duration.
        # Over 4 wins if the last digit of the final tick > 4
        # Under 5 wins if the last digit of the final tick < 5
        # CRITICAL: Use the correct decimal places for digit extraction!
        final_tick = all_ticks[i + look_ahead]
        final_price_str = f"{final_tick['quote']:.{dp}f}"
        final_digit = int(final_price_str[-1])
        
        # Label: 1 = Over hits (last digit > barrier), 0 = Under
        label = 1 if final_digit > OVER_BARRIER else 0
        
        features_list.append(features)
        labels_list.append(label)
    
    logger.info(f"Training data: {len(features_list)} samples")
    
    # Log class balance
    if labels_list:
        pos_rate = sum(labels_list) / len(labels_list)
        logger.info(f"Class balance: {pos_rate:.1%} Over, {1-pos_rate:.1%} Under")
    
    return features_list, labels_list


def train_model(features_list: list, labels_list: list,
                model_type: str = "logistic") -> OverUnderModel:
    """
    Train the model on collected data.
    Uses walk-forward validation for realistic accuracy estimate.
    """
    model = OverUnderModel(model_type=model_type)
    
    if not features_list:
        logger.error("No training data!")
        return model
    
    # Walk-forward split: 70% train, 30% test
    split_idx = int(len(features_list) * 0.7)
    train_features = features_list[:split_idx]
    train_labels = labels_list[:split_idx]
    test_features = features_list[split_idx:]
    test_labels = labels_list[split_idx:]
    
    # Train on training set
    logger.info(f"Training on {len(train_features)} samples...")
    model.warmup(train_features, train_labels)
    
    # Evaluate on test set (out-of-sample)
    correct = 0
    total = len(test_features)
    
    for features, label in zip(test_features, test_labels):
        pred = model.predict(features)
        if pred.predicted_class == label:
            correct += 1
    
    test_acc = correct / total if total > 0 else 0
    logger.info(f"Out-of-sample accuracy: {test_acc:.1%} ({correct}/{total})")
    
    if test_acc < 0.52:
        logger.warning(
            f"⚠️  Out-of-sample accuracy ({test_acc:.1%}) is below 52%. "
            f"The model may not have a real edge. Consider:"
            f"\n  - Collecting more data"
            f"\n  - Adding more features"
            f"\n  - Trying a different model type"
            f"\n  - NOT going live yet"
        )
    elif test_acc < 0.56:
        logger.warning(
            f"Out-of-sample accuracy ({test_acc:.1%}) is marginal. "
            f"Proceed with caution — paper trade first."
        )
    else:
        logger.info(f"✅ Out-of-sample accuracy ({test_acc:.1%}) looks promising!")
    
    # Now train on ALL data for the final model
    logger.info("Training final model on all data...")
    final_model = OverUnderModel(model_type=model_type)
    final_model.warmup(features_list, labels_list)
    
    return final_model


async def main_async():
    parser = argparse.ArgumentParser(description="Warmup trainer")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--model", default="logistic",
                        choices=["logistic", "forest", "boosting"])
    args = parser.parse_args()
    
    # Step 1: Collect data
    features, labels = await collect_training_data(args.symbol, args.samples)
    
    if not features:
        logger.error("No training data collected. Exiting.")
        return
    
    # Step 2: Train model
    model = train_model(features, labels, model_type=args.model)
    
    # Step 3: Save snapshot
    from models.model_persistence import ModelPersistence
    persistence = ModelPersistence()
    path = persistence.save_snapshot(model, snapshot_name="warmup_model")
    
    if path:
        logger.info(f"Model saved to: {path}")
        logger.info("You can now run the bot with: python main.py")
    
    # Print feature importance (logistic only)
    if args.model == "logistic":
        importance = model.get_feature_importance()
        if importance:
            logger.info("\nTop 10 Feature Weights:")
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, weight in sorted_features[:10]:
                logger.info(f"  {name:35s} {weight:+.4f}")


if __name__ == "__main__":
    asyncio.run(main_async())
