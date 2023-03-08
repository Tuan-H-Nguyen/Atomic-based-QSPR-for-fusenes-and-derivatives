import pickle, glob
import numpy as np

all_files = glob.glob("*.pkl")

data = ["mixed","pah","subst"]
method = ["shortest_path","subtree","edge"]

keys = [[m,d] for d in data for m in method]

def exam_list(list_):
    if type(list_) == list:
        print("list w len:",len(list_))
        for i in list_:
            exam_list(i)

    elif type(list_) == np.ndarray:
        print("np.array w shape:",list_.shape)

    else:
        print(type(list_))

for key in keys:
    print(">>>",key)
    combined_result = {
        "train_set_size": [],
        "active":[],
        "random":[]
        }

    for file in all_files:
        if "experiments" not in file or key[0] not in file or key[1] not in file:
            continue

        print(file)

        with open(file,"rb") as handle:
            result = pickle.load(handle)

        for k,v in result.items():
            if k == "train_set_size":
                combined_result[k] = v
            else:
                combined_result[k] += v[0]
        
        print(np.array(combined_result["active"]).shape)

        with open("active_learning_"+key[0]+"_"+key[1]+".pkl","wb") as log:
            pickle.dump(combined_result,log)


