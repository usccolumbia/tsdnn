import csv

with open('data_test.csv') as f:
    reader = csv.reader(f)
    test = [row[0] for row in reader]

with open('data_unlabeled_full.csv') as f:
    reader = csv.reader(f)
    u = [row[0] for row in reader]

with open('data_positive.csv') as f:
    reader = csv.reader(f)
    pos = [row[0] for row in reader]

for i, cid in enumerate(test):
    if cid in pos:
        print('INTRUDER')
    # if cid in pos:
    #     pos.remove(cid)
    # elif cid in u:
    #     u.remove(cid)
    # print(i)

# new_pos = []
# for cid in pos:
#     new_pos.append([cid])

# new_unl = []
# for cid in u:
#     new_unl.append([cid])

# with open('new_positive.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(new_pos)

# with open('new_unlabeled.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(new_unl)

