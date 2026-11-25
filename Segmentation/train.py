"""Corrected Auto-DeepLab retraining entry point.

See BASELINE.md and run_retrain_baseline.sh. The earlier PPO checkpoint format
is evaluation-only in this tree; it cannot resume this corrected architecture.
"""
from train_baseline import main

if __name__ == '__main__':
    main()
