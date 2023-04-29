
from PIL import Image
import os

from pathlib import PureWindowsPath, PurePosixPath,Path

import pandas as pd


DATA_DIR ="D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/img/"

new = Image.new(mode="RGB", size=(256,256))
# new.save('./gg.jpg')

def save_img(root1, label, new, file):
    label_src_root = root1.replace('img', label)
    if not os.path.exists(label_src_root):
        os.makedirs(label_src_root)
    label_src = os.path.join(label_src_root,file)
    new.save(label_src)

def create_empty_images():
    for (root,dirs,files) in os.walk(DATA_DIR, topdown=True):
        for file in files:
            root1=str(PurePosixPath(PureWindowsPath(root)))
            root1 =root1.replace('D:\/', 'D:/')
            print(root1)
            file = file.replace('jpg', 'png')
            save_img(root1, 'labels',new, file)
            save_img(root1, 'edges',new, file)
            save_img(root1, 'labels_rev',new, file)

def path_replace(root1, label, file):
    label_src_root = root1.replace('img', label)
    return os.path.join(label_src_root,file)

def get_rel_path(root_path, idx_name, file): 
    src = path_replace(root_path, idx_name, file)  
    path = Path(src)
    index = path.parts.index(idx_name)
    return  "/".join(path.parts[index:])


def get_data():
    img_path_list,label_path_list, edge_path_list,f_name_list = [] ,[],[],[]
    out_path_list = []
    out_root = 'D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/output/cihp_parsing_maps/'

    for (root,dirs,files) in os.walk(DATA_DIR, topdown=True):
        for file in files[:10]:
            root1=str(PurePosixPath(PureWindowsPath(root)))
            root1 =root1.replace('D:\/', 'D:/')

            img_root= get_rel_path(root1, 'img', '')
            out_path = os.path.join(out_root,img_root)
            out_path_list.append(out_path)

            img_path= get_rel_path(root1, 'img', file)
            img_path_list.append(img_path)

            file = file.replace('jpg', 'png')
            label_path= get_rel_path(root1, 'labels', file)
            label_path_list.append(label_path)
            edge_path= get_rel_path(root1, 'edges', file)
            edge_path_list.append(edge_path)
            f_name = file.split(".")[0]
            f_name_list.append(f_name)
    return img_path_list,label_path_list, edge_path_list, f_name_list,out_path_list

img_path_list,label_path_list, edge_path_list,f_name_list,out_path_list = get_data()
df = pd.DataFrame({"img":img_path_list, "label": label_path_list, "edges": edge_path_list, "f_name": f_name_list, "out_root": out_path_list, })

val='val'
val_path = f'D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\datasets\CIHP\list\{val}.csv'
val_path=str(PurePosixPath(PureWindowsPath(val_path)))
val_path =val_path.replace('D:\/', 'D:/')
print(val_path)

df.to_csv(val_path)

# with open(val_path, 'a') as f:
#     dfAsString = df[['img','label']].to_string(header=False, index=False)
#     f.write(dfAsString)

# val_id='val_id'
# val_id_path = f'D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\datasets\CIHP\list\{val_id}.txt'
# val_id_path=str(PurePosixPath(PureWindowsPath(val_id_path)))
# val_id_path =val_path.replace('D:\/', 'D:/')
# with open(val_id_path, 'a') as f1:
#     dfAsString1 = df[['f_name']].to_string(header=False, index=False)
#     f1.write(dfAsString1)
    
# p='D:/LjmuMSc/Projects/Github/Relevent_repos/CIHP_PGN-master/CIHP_PGN-master/datasets/CIHP/img/WOMEN/Tees_Tanks/id_00007976/thy34.jpg'
# print(get_rel_path(p, '/img'))
# 

# out_path_list = []
# for (root,dirs,files) in os.walk(DATA_DIR, topdown=True):
#         for file in files[:10]:
#             root1=str(PurePosixPath(PureWindowsPath(root)))
#             root1 =root1.replace('D:\/', 'D:/')
#             img_root= get_rel_path(root1, 'img', '')
           
#             out_path = os.path.join(out_root,img_root)
#             print(out_path)
#             out_path_list.append(out_path)

# df = pd.DataFrame({"out_path":out_path_list})

# val='out_path'
# val_path = f'D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\datasets\CIHP\list\{val}.txt'
# val_path=str(PurePosixPath(PureWindowsPath(val_path)))
# val_path =val_path.replace('D:\/', 'D:/')
# print(val_path)
# with open(val_path, 'a') as f:
#     dfAsString = df[['out_path']].to_string(header=False, index=False)
#     f.write(dfAsString)


# val='out_path'
# val_path = f'D:\LjmuMSc\Projects\Github\Relevent_repos\CIHP_PGN-master\CIHP_PGN-master\datasets\CIHP\list\{val}.txt'
# val_path=str(PurePosixPath(PureWindowsPath(val_path)))
# val_path =val_path.replace('D:\/', 'D:/')

# out_list = []
# with open(val_path, 'r') as f:
#     out_list = f.readlines()

# parsing_dir =out_list[10].strip()
# if not os.path.exists(parsing_dir):
#     os.makedirs(parsing_dir)
