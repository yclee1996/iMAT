import numpy as np

#######################################
def str_list2float_list(str_l):
    return_list = []
    for x in str_l:
        return_list.append(float(x))
    return return_list
#######################################
def get_coord(file_target):
   
    coord_2 = []
    coord_3 = []
    with open(file_target, "r") as p:
        for i, line in enumerate(p):
            coord_1 = []
            if "#" not in line:
                #d = str_list2float_list(line.split())[0:3]
                b = float(line.split()[0])
                e = float(line.split()[3])
                f = float(line.split()[1]) 
                coord_1.append(b)
                coord_1.append(e)
                coord_3.append(f)
                coord_2 += coord_1
                a = int(len(coord_2)/2)
    all_coord_1 = np.array(coord_2).reshape(a,2)
    all_coord_2 = np.array(coord_3).reshape(a,1)
    return all_coord_1, all_coord_2
######################################
inp, y = get_coord("vector_vs_energy")
print (inp)
print (y)

