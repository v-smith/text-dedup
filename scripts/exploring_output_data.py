from datasets import Dataset
import pandas as pd

ds = Dataset.from_file("../output/minhash/data-00000-of-00001.arrow")

df = pd.DataFrame(ds)


a = 1
