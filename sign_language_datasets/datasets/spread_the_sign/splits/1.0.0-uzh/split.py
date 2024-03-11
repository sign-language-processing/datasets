import random
import sys

import pandas as pd


seed = 3407
csv_path = sys.argv[1]
out_path = sys.argv[2]

def write(filename, video_ids):
    with open(out_path + filename, 'w') as f:
        for line in video_ids:
            f.write(f"{line}\n")

df = pd.read_csv(csv_path)
video_ids = df.index.values.tolist()

write('all.txt', video_ids)

random.seed(seed)
random.shuffle(video_ids)

length = len(video_ids)
val_ratio = 0.001
val_idx = int(length * val_ratio)
test_ratio = 0.001
test_idx = val_idx + int(length * test_ratio)

video_ids_val = video_ids[:val_idx]
video_ids_test = video_ids[val_idx:test_idx]
video_ids_train = video_ids[test_idx:]

write('train.txt', video_ids_train)
write('val.txt', video_ids_val)
write('test.txt', video_ids_test)