"""Entry point for corrected, from-scratch LoRA/CVaR cell search.

The legacy dense DARTS/PPO implementation is replaced by the workflow in
train_search_corrected.py. See SEARCH.md for changed search scope and arguments.
"""
from train_search_corrected import main

if __name__ == '__main__':
    main()
