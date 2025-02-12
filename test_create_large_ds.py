import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets

# create large tsv files for testing
def create_large_tsv_files(n_datasets=1000):
    # create large tsv files
    for i in range(1, n_datasets + 1):
        df = pd.DataFrame(np.random.randint(0, 100, size=(300, 444)))
        df.to_csv(f'./data/large_{i}.tsv', sep='\t', index=False)

# convert tsv to huggingface dataset
def convert_tsv_to_huggingface_dataset(n_datasets=1000):
    # create large tsv files
    datasets = []
    for i in range(1, n_datasets + 1):
        df = pd.read_csv(f'./data/large_{i}.tsv', sep='\t')
        dataset = Dataset.from_pandas(df)
        datasets.append(dataset)
    concatenated_dataset = concatenate_datasets(datasets)
    concatenated_dataset.save_to_disk('./data/large_concatenated')


if __name__ == '__main__':
    n_datasets = 1000
    # create_large_tsv_files(n_datasets)
    convert_tsv_to_huggingface_dataset(n_datasets)
