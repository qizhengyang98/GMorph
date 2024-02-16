import bisect
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DatasetSampler:
    def __init__(
        self,
        datasets: List[Dataset],
        models: List[nn.Module],
        device: str,
        keep_sizes=Optional[List[int]],
        batch_size=128,
        rand_seed=None,
        num_workers=4,
        use_transformer=False,
        glue_task=False,
        preprocess_func=None
    ):
        if rand_seed:
            torch.manual_seed(rand_seed)
        # Init number of data to keep in each dataset
        if keep_sizes is None:
            # keep all data if keep_sizes is not defined
            keep_sizes = [len(dataset) for dataset in datasets]
        else:
            # Check if the sizes are aligned with the # of datasets
            assert len(keep_sizes) == len(datasets)
        
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_input = True if glue_task else False

        # Assemble new dataset
        self.datasets = self.make_data(datasets, keep_sizes)
        self.targets = self.generate_labels(models, use_transformer, glue_task, preprocess_func)
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx: int):
        if self.text_input:
            return self.datasets[idx]['sentence'], self.get_dataset_index(idx), [target[idx] for target in self.targets]
        else:
            return self.datasets[idx][0], self.get_dataset_index(idx), [target[idx] for target in self.targets]
    
    def get_dataset_index(self, idx: int):
        if idx < 0:
            if -idx > len(self.datasets):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self.datasets) + idx
        dataset_idx = bisect.bisect_right(self.datasets.cumulative_sizes, idx)
        return dataset_idx

    def make_data(self, datasets: List[Dataset], keep_sizes: List[int]) -> Dataset:
        sampled_datasets = []
        for i, dataset in enumerate(datasets):
            keep_size = keep_sizes[i]
            discard_size = len(dataset) - keep_size
            keep_dataset, _ = torch.utils.data.random_split(dataset, [keep_size, discard_size])
            sampled_datasets.append(keep_dataset)
        return torch.utils.data.ConcatDataset(sampled_datasets)
    
    def generate_labels(self, models: List[nn.Module], use_transformer=False, glue_task=False, preprocess_func=None) -> List[torch.Tensor]:
        dataloader = torch.utils.data.DataLoader(
            self.datasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        for model in models:
            model.eval()
        
        targets = [[] for _ in models]
        # start, end = 0, self.batch_size
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if glue_task:
                    data = preprocess_func(batch['sentence']).to(self.device)
                else:
                    data = batch[0].to(self.device)
                # end = len(data) + start
                for j, model in enumerate(models):
                    model = model.to(self.device)
                    if use_transformer:
                        if glue_task:
                            outputs = model(**data).logits.detach().cpu()
                        else:
                            outputs = model(data).logits.detach().cpu()
                    else:
                        outputs = model(data).detach().cpu()
                    # labels = torch.argmax(model(data), dim=1)
                    targets[j].extend(outputs)
                # start = end
        return targets

    def train_test_split(self, train_percent: float) -> Tuple[Dataset, Dataset]:
        assert 0 < train_percent <= 1
        train_size = int(train_percent * len(self.datasets))
        test_size = len(self.datasets) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.datasets, [train_size, test_size])
        
        return train_dataset, test_dataset
