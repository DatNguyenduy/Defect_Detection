import os



def get_all_item_label_path(input_dir:str,item_suffix='.jpg') -> list:
    list_item_paths = []
    list_label_paths = []
    for root,dirs,files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1] == item_suffix:
                list_item_paths.append(os.path.join(root,file))
                file_label = os.path.splitext(file)[0] + "_label.bmp"
                list_label_paths.append(os.path.join(root,file_label))
    list_item_paths = sorted(list_item_paths)
    list_label_paths = sorted(list_label_paths)
    return list_item_paths,list_label_paths

