#!/bin/bash

# Define the base directory where your zip files are located
BASE_DIR="data/Atari-HEAD"

echo Downloading Atari-HEAD
curl -L https://zenodo.org/api/records/3451402/files-archive -o head.zip
unzip head.zip -d $BASE_DIR
rm head.zip

# Loop through each .zip file in the specified directory
for zip_file_path in "$BASE_DIR"/*.zip; do
  # Unzip the contents into the new directory
  unzip "$zip_file_path" -d "$BASE_DIR"
  
  # Delete the original zip file if extraction was successful
  if [ $? -eq 0 ]; then # Check if the last command (unzip) was successful
    rm "$zip_file_path"
  else
    echo "Error unzipping $zip_file_path. Not deleting original."
  fi

  # Untar all the trial files
  game_name=$(basename "$zip_file_path")
  game_name="${game_name%.zip}"
  for trial_tar in "$BASE_DIR/$game_name"/*.tar.bz2; do
    echo $trial_tar
    tar -xjf "$trial_tar" -C "$BASE_DIR/$game_name"

    # Delete the original zip file if extraction was successful
    if [ $? -eq 0 ]; then # Check if the last command (unzip) was successful
        rm "$trial_tar"
    else
        echo "Error unzipping $trial_tar. Not deleting original."
    fi
  done
done

# Tranform .txt files to .csv
game_dirs="["
for game_dir in "$BASE_DIR"/*; do
  if [ ! -d $game_dir ]; then
    continue
  fi
  game_dirs+="'$game_dir',"
done
game_dirs=${game_dirs::-1}
game_dirs+="]"
echo $game_dirs
python -c "from atari_cr.atari_head.utils import transform_to_proper_csv; [transform_to_proper_csv(name) for name in $game_dirs]"