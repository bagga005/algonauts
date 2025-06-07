#!/bin/bash

# Check if argument is provided
if [ -z "$1" ]; then
  echo "Error: STRATEGY_FOLDER argument not provided."
  echo "Usage: $0 <STRATEGY_FOLDER>"
  exit 1
fi

# Get the first argument as the EMBEDDINGS_PATH
STRATEGY_FOLDER="$1"

#set embeddings folder
EMBEDDINGS_COMBINED_FOLDER="combined_embeddings3"

# Loop through the rest of the arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v1)
      EMBEDDINGS_COMBINED_FOLDER="combined_embeddings8"
      echo "Using embeddings folder: $EMBEDDINGS_COMBINED_FOLDER for v1"
      shift
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

EMBEDDINGS_COMBINED_PATH="/workspace/algo_data/$EMBEDDINGS_COMBINED_FOLDER"
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
