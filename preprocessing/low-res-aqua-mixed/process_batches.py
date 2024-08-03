"""
process_batches.py

Processes the parquet files for lowres and lowres aqua into batches of 1000 rows in Torch format.
Files specified use the same format/columns as in the train data.
"""

import os
import polars as pl
import torch

def process_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    #Get all the file names in the input direction
    all_files = [
        os.path.join(dirname, filename)
        for dirname, _, filenames in os.walk(input_dir)
        for filename in filenames
        if 'parquet' in filename
    ]

    pt_num = 0
    index_num = 0

    #Loop through all the files
    for f in all_files:
        try:
            df = pl.read_parquet(f).drop('sample_id') #Read the parquet file into RAM as a polars dataframe
            os.remove(f) #Delete the original file to save HD space
            print(f"Removing {f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

        #Get the 1000 rows slices of the file at a time
        for idx, frame in enumerate(df.iter_slices(n_rows=1000)):
            data = frame.to_torch('tensor', dtype=pl.Float64) #Make it into a torch tensor
            if data.shape[0] == 1000: #only save the tensor if it contains a full 1000 datapoints
                torch.save(data, f'{output_dir}/{idx}{index_num}.pt')
                index_num += 1
                pt_num += 1
                if pt_num % 1000 == 0:
                    print(pt_num)

#Process both the low res and aqua planet datasets
process_files('./raw_lowres/', './lowres_torch')
process_files('./raw_aqua/', './ocean_torch')
