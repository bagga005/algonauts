#!/bin/bash
set -e  # Exit immediately if any command fails

# Check if all 3 arguments are provided
if [ $# -ne 3 ]; then
  echo "Error: All 3 arguments must be provided."
  echo "Usage: $0 <STRATEGY_FOLDER> <COMBINED_FOLDER> <ONLY_250>"
  echo "  STRATEGY_FOLDER: The strategy folder name"
  echo "  COMBINED_FOLDER: The combined folder identifier"
  echo "  ONLY_250: Set to 'True' to run only 250 features, any other value runs all sizes"
  exit 1
fi

# Get the first argument as the EMBEDDINGS_PATH
STRATEGY_FOLDER="$1"
COMBINED_FOLDER="$2"
ONLY_250="$3"


#set embeddings folder
EMBEDDINGS_COMBINED_FOLDER="combined_embeddings"$COMBINED_FOLDER
echo "using combined folder $EMBEDDINGS_COMBINED_FOLDER"


EMBEDDINGS_COMBINED_PATH="/workspace/algo_data/$EMBEDDINGS_COMBINED_FOLDER"
DESTINATION_PATH="/workspace/stimulus_features/pca/friends_movie10/visual/features_train.npy"


cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-250.npy" "$DESTINATION_PATH"
echo "******Doing RUN FOR 250"
python run_experiements.py $STRATEGY_FOLDER-250 $EMBEDDINGS_COMBINED_PATH/$RESULTS_FOLDER

RESULTS_FOLDER="evals"

if [ "$ONLY_250" != "True" ]; then
  cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-500.npy" "$DESTINATION_PATH"
  echo "******Doing RUN FOR 500"
  python run_experiements.py $STRATEGY_FOLDER-500 $EMBEDDINGS_COMBINED_PATH/$RESULTS_FOLDER
  cp "$EMBEDDINGS_COMBINED_PATH/$STRATEGY_FOLDER/features_train-1000.npy" "$DESTINATION_PATH"
  echo "******Doing RUN FOR 1000"
  python run_experiements.py $STRATEGY_FOLDER-1000 $EMBEDDINGS_COMBINED_PATH/$RESULTS_FOLDER
fi