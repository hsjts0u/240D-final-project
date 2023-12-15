import json
import sys

n = len(sys.argv)

for i in range(1,n):
    acc = []
    f = open(sys.argv[i])
    data = json.load(f)
    for j in data["results"]:
        acc.append(data["results"][j]["acc"])

    acc_mean = sum(acc) / len(acc)
    print(acc_mean)
    f.close()