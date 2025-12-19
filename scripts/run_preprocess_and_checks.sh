#!/bin/bash

# Clean up old processed files
rm -rf data/processed
rm -rf processed_graphs

# Run preprocess
python scripts/preprocess_transactions_improved.py --input data/raw/transactions.csv --out data/raw/processed.csv --encoders data/encoders/encoders.pkl

# Run feature combination analyzer
python analysis/feature_combo_stats.py --config-name config dataset=transaction

# Run debug training
python src/main.py --config-name config dataset=transaction +general.debug=true general.name=debug_preproc
