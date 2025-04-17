import pandas as pd
import torch

def extract_features_from_row(row):
    numbers = row[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values.astype(int)
    one_digit = sum(1 <= n <= 9 for n in numbers)
    twenty = sum(20 <= n <= 29 for n in numbers)
    thirty = sum(30 <= n <= 39 for n in numbers)
    forty = sum(40 <= n <= 45 for n in numbers)
    has_consecutive = int(any(b - a == 1 for a, b in zip(sorted(numbers), sorted(numbers)[1:])))
    bonus = row['bonus']
    return [one_digit, twenty, thirty, forty, has_consecutive, bonus]

def make_input_tensor_from_csv(csv_path, seq_len=10):
    df = pd.read_csv(csv_path)
    df = df.sort_values('회차')

    features = df.apply(extract_features_from_row, axis=1).tolist()
    features = torch.tensor(features, dtype=torch.float32)

    data = []
    for i in range(len(features) - seq_len):
        window = features[i:i + seq_len]
        data.append(window)

    return torch.stack(data)  # [batch, seq_len, input_dim]


csv_path = "lotto_recent30.csv" 
tensor_data = make_input_tensor_from_csv(csv_path)
print(tensor_data.shape)  # torch.Size([N, 10, 6])
