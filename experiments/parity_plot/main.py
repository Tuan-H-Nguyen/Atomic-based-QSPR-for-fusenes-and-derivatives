import sys, os
path = os.path.dirname(os.path.realpath(__file__)).split("\\")
print("\\".join(path[:-2]))
sys.path.append("\\".join(path[:-2]))


