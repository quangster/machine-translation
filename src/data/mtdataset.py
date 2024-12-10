import torch
from torch.utils.data import Dataset

class MTDataset(Dataset):
    def __init__(self, inputs: list[int], outputs: list[int], max_length: int=10, padding_idx: int=1):
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.build_dataset(inputs, outputs)
    
    def build_dataset(self, inputs, outputs):
        self.inputs = []
        self.outputs = []
        for input, output in zip(inputs, outputs):
            if len(input) > self.max_length or len(output) > self.max_length:
                continue
            if len(input) == 0 or len(output) == 0:
                continue

            self.inputs.append(self.convert_to_tensor(input))
            self.outputs.append(self.convert_to_tensor(output))
    
    def convert_to_tensor(self, indexes: list[int]) -> torch.Tensor:
        indexes = indexes + [self.padding_idx] * (self.max_length - len(indexes))
        return torch.tensor(indexes, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]