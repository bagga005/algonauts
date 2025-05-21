#!/bin/bash

# Check if argument is provided
if [ -z "$1" ]; then
  echo "Error: STRATEGY_FOLDER argument not provided."
  echo "Usage: $0 <STRATEGY_FOLDER>"
  exit 1
fi

# Get the first argument as the EMBEDDINGS_PATH
STRATEGY_FOLDER="$1"
EMBEDDINGS_COMBINED_PATH="/workspace/algo_data/embeddings_combined"
DESTINATION_PATH="/workspace/stimulus_features/pca/friends_movie10/visual/features_train.npy"


cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-250.npy" "$DESTINATION_PATH"
echo "******Doing RUN FOR 250"
python run_experiements.py
cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-500.npy" "$DESTINATION_PATH"
echo "******Doing RUN FOR 500"
python run_experiements.py
cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-1000.npy" "$DESTINATION_PATH"
echo "******Doing RUN FOR 1000"
python run_experiements.py