import numpy as np
import pandas as pd
import random
from tqdm import tqdm 
from random import randint

df = pd.read_csv("data.tsv", sep="\t")   # read dummy .tsv file into memory

train_data = df.values  # access the numpy array containing values
print("Loaded data")

final_data = []
temp  = []
for i in tqdm(train_data):
	if len(temp) == 0 and i[3] != 1:
		temp.append(i)
	elif i[0] == temp[-1][0] and i[3] != 1:
		temp.append(i)
	elif i[3] == 1 and (i[0] == temp[-1][0] or len(temp) == 0):
		final_data.append(i)
	else:
		number = randint(0, len(temp)-1)
		indices = random.sample(range(0, len(temp)), number)
		for j in indices:
			final_data.append(temp[j])
		temp = []
		if i[3] == 1:
			final_data.append(i)
		else:
			temp.append(i)

if len(temp) != 0:
	number = randint(0, len(temp)-1)
	indices = random.sample(range(0, len(temp)), number)
	for i in indices:
		final_data.append(temp[i])

np.savetxt("train_data_random.tsv", final_data, delimiter="\t", fmt='%i\t%s\t%s\t%i\t%i')