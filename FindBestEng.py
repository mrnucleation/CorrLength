import sys

try: filename = sys.argv[1]
except IndexError: filename = "dumpfile.dat"



with open(filename, "r") as infile, open("score.test", "w") as outfile:
    minval_train = 1e300
    coords_train = ""
    cnt = 0
    minloc = 0
    e0 = 1e300
    for i, line in enumerate(infile):
        cnt += 1
        col = line.split("|")
        eng_train = float(col[-1])
#        e0_train = float(col[-2])
        if minval_train > eng_train:
            minval_train = eng_train
            minloc_train = i+1
            coords_train = col[0]
#            e0 = e0_train
        outfile.write("%s %s\n"%(i+1, minval_train))

print("Lowest Train Value: %s"%(minval_train))
with open("Parameters.dat", "w") as outfile:
    for x in [float(x) for x in coords_train.split()]:
        outfile.write("%s\n"%(x))
print("Number of Total Evaluations: %s"%(cnt))
print("Number of 500K Correlation Length: %s"%(e0))
print("Number of Evaluations till Train Minima was found: %s"%(minloc_train))
