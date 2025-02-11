import os
import torch
import torch.utils.data as data
import pandas as pd

class HistopathologyDataset(data.Dataset):
    def __init__(self, embeddings_dir, labels_csv, transform=None):
        """
        Args:
            embeddings_dir (str): Directory where .pt embedding files are stored.
            labels_csv (str): CSV file with columns [slide_id, TP53] (or TP53_mutation).
                              The 'slide_id' column contains a partial name (without ".pt")
                              that appears in the .pt file names.
            transform (callable, optional): Optional transform to be applied on a bag.
        """
        self.embeddings_dir = embeddings_dir
        # Read CSV file
        df = pd.read_csv(labels_csv)
        self.transform = transform
        # Get sorted list of all .pt files for deterministic matching
        self.embedding_files = sorted([f for f in os.listdir(self.embeddings_dir) if f.endswith('.pt')])
        
        # Filter out CSV rows that have no matching .pt file
        valid_rows = []
        for idx, row in df.iterrows():
            slide_partial = row['Tumor_Sample_Barcode']
            matching_files = [fname for fname in self.embedding_files if slide_partial in fname]
            if len(matching_files) == 0:
                print(f"Warning: No matching .pt file found for slide id partial: '{slide_partial}'. Skipping this entry.")
            else:
                valid_rows.append(row)
        if len(valid_rows) == 0:
            raise RuntimeError("No valid entries found in the CSV file. Please check your file names.")
        self.labels_df = pd.DataFrame(valid_rows)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index):
        row = self.labels_df.iloc[index]
        slide_partial = row['Tumor_Sample_Barcode']  # partial name from CSV
        label = row['TP53']              # or 'TP53_mutation', depending on your column name

        # Find all matching files in the sorted list
        matching_files = [fname for fname in self.embedding_files if slide_partial in fname]
        if len(matching_files) > 1:
            print(f"Warning: Multiple matches found for slide id partial: '{slide_partial}'. Combining {len(matching_files)} files.")
        
        bags = []
        for file_to_load in matching_files:
            bag = torch.load(os.path.join(self.embeddings_dir, file_to_load))
            # If bag is stored as [1024, varied] and needs to be [varied, 1024], then transpose it.
            if bag.ndim == 2 and bag.shape[0] == 1024 and bag.shape[1] != 1024:
                bag = bag.transpose(0, 1)
            bags.append(bag)
        # Concatenate along the instance (row) dimension
        bag = torch.cat(bags, dim=0)
    
        if self.transform:
            bag = self.transform(bag)
    
        return bag, torch.tensor([label], dtype=torch.float32)
