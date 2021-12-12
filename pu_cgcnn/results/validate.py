import os
import csv
from operator import truediv

directory = 'validation'

ids = []
preds = []
counter = []

for f, filename in enumerate(os.listdir(directory)):
    print(f)
    with open(directory + '/' + filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] not in ids:
                ids.append(row[0])
                preds.append(float(row[1]))
                counter.append(1)
            else:
                i = ids.index(row[0])
                preds[i] += float(row[1])
                counter[i] += 1

avg_preds = list(map(truediv, preds, counter))
with open('avg_' + directory + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ids', 'predictions'])
    writer.writerows(list(zip(ids, avg_preds)))

true_pos = sum(i >= 0.5 for i in avg_preds) / len(avg_preds)
print(f'True Positive Rate: {true_pos}')
