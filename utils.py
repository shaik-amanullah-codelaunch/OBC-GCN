import random
import numpy as np
from scipy import sparse
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GA_ReadOnly

# Auhui Map (for example)
category_map = {
    'building': [1, 0, 0, 0, 0],
    'bare': [0, 1, 0, 0, 0],
    'road': [0, 0, 1, 0, 0],
    'vegetation': [0, 0, 0, 1, 0],
    'water': [0, 0, 0, 0, 1]
}

# # def load_data(mask_path, features_path, training_data_path, validation_data_path, test_data_path):
# #     # Load mask data using GDAL
#     src_ds = gdal.Open(mask_path, GA_ReadOnly)
#     objects_img = gdal_array.DatasetReadAsArray(src_ds)
#     count = np.max(objects_img) + 1  # Number of unique objects in the mask

#     # Load features data
#     features_dim = 181  # Assuming 181 feature dimensions
#     features = np.zeros((count, features_dim), 'float32')
    
#     with open(features_path) as file:
#         for line in file.readlines():
#             splited_line = line.split('\t')
#             object_id = int(splited_line[0])
#             features[object_id][:] = [float(feature) for feature in splited_line[1:]]
    
#     features = sparse.lil_matrix(features, dtype='float32')

#     # Build adjacency matrix
#     adjency = np.zeros((count, count), dtype='float32')
    
#     for row_id, row in enumerate(objects_img):
#         if (row_id + 1) < src_ds.RasterYSize:
#             for col_id, dn in enumerate(row):
#                 if (col_id + 1) < src_ds.RasterXSize:
#                     right_dn = objects_img[row_id, col_id + 1]
#                     down_dn = objects_img[row_id + 1, col_id]
#                     right_down_dn = objects_img[row_id + 1, col_id + 1]
                    
#                     if dn != right_dn:
#                         adjency[dn, right_dn] = adjency[right_dn, dn] = 1
#                     if dn != down_dn:
#                         adjency[dn, down_dn] = adjency[down_dn, dn] = 1
#                     if dn != right_down_dn:
#                         adjency[dn, right_down_dn] = adjency[right_down_dn, dn] = 1

#                 if col_id > 0:
#                     left_down_dn = objects_img[row_id + 1, col_id - 1]
#                     if dn != down_dn:
#                         adjency[dn, left_down_dn] = adjency[left_down_dn, dn] = 1

#     out_adj = sparse.lil_matrix(adjency, dtype='float32')

#     # Load training, validation, and test data
#     category_dim = len(category_map)
#     y_train, train_mask = _load_labels_and_mask(training_data_path, count, category_dim)
#     y_val, val_mask = _load_labels_and_mask(validation_data_path, count, category_dim)
#     y_test, test_mask = _load_labels_and_mask(test_data_path, count, category_dim)

#     return out_adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask


def load_data(training_data_path, validation_data_path, test_data_path):
    # Dummy count
    count = 10000  # Set this to a higher value based on your dataset
    
    # Generate dummy features
    features_dim = 181  # Adjust if necessary
    features = np.random.rand(count, features_dim).astype('float32')
    features = sparse.lil_matrix(features, dtype='float32')
    
    # Dummy adjacency matrix
    adj = sparse.lil_matrix((count, count), dtype='float32')  # Adjust if needed
    
    # Load train, validation, and test data
    def load_mask(file_path):
        category_dim = len(category_map)
        y = np.zeros((count, category_dim))
        mask = [False] * count
        with open(file_path) as file:
            for line in file.readlines():
                splited_line = line.split('\n')[0].split('\t')
                object_id = int(splited_line[0])
                category = category_map.get(splited_line[1], [0]*category_dim)
                
                # Ensure object_id is within bounds
                if object_id >= count:
                    continue  # Skip out-of-bounds indices
                
                y[object_id] = np.array(category, dtype='int')
                mask[object_id] = True
        return y, np.array(mask, dtype='bool')
    
    y_train, train_mask = load_mask(training_data_path)
    y_val, val_mask = load_mask(validation_data_path)
    y_test, test_mask = load_mask(test_data_path)

    return adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask

def generate_npz(path):
    adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask = load_data(
        "data/ah/train.txt", 
        "data/ah/val.txt", 
        "data/ah/test.txt"
    )
    np.savez(path, adj=adj, features=features, y_train=y_train, train_mask=train_mask, y_val=y_val, val_mask=val_mask, y_test=y_test, test_mask=test_mask)

if __name__ == '__main__':
    generate_npz('data/test.npz')
def _load_labels_and_mask(data_path, count, category_dim):
    y_data = np.zeros((count, category_dim))
    mask = [False] * count
    
    with open(data_path) as file:
        for line in file.readlines():
            splited_line = line.split('\n')[0].split('\t')
            object_id = int(splited_line[0])
            category = category_map[splited_line[1]]
            y_data[object_id] = np.array(category, dtype='int')
            mask[object_id] = True

    return y_data, np.array(mask, dtype='bool')


def generate_npz(path):
    adj, features, y_train, train_mask, y_val, val_mask, y_test, test_mask = load_data(
        "data/ah/train.txt", "data/ah/val.txt", "data/ah/test.txt"
    )
    np.savez(path, adj=adj, features=features, y_train=y_train, train_mask=train_mask, 
             y_val=y_val, val_mask=val_mask, y_test=y_test, test_mask=test_mask)


def generate_sub_npz(input_path, output_path, sub_count):
    data = np.load(input_path)
    adj = data['adj'][()]
    features = data['features'][()]
    y_train = data['y_train'][()]
    train_mask = data['train_mask'][()]
    y_val = data['y_val'][()]
    val_mask = data['val_mask'][()]

    train_ids = np.where(train_mask)[0]  # Get the indices of training samples
    sub_train_ids = random.sample(list(train_ids), sub_count)

    # Reset the train_mask to only include the selected subset
    train_mask[:] = False
    train_mask[sub_train_ids] = True

    np.savez(output_path, adj=adj, features=features, y_train=y_train, train_mask=train_mask, 
             y_val=y_val, val_mask=val_mask)


if __name__ == '__main__':
    generate_npz('data/test.npz')

    data = np.load('data/test.npz', allow_pickle=True)
    adj = data['adj'][()]
    features = data['features'][()]
    y_train = data['y_train'][()]
    train_mask = data['train_mask'][()]
    y_val = data['y_val'][()]
    val_mask = data['val_mask'][()]
    y_test = data['y_test'][()]
    test_mask = data['test_mask'][()]

    print(repr(adj))
    print(repr(features))
    print(repr(y_train))
    print(repr(train_mask))
    print(repr(y_val))
    print(repr(val_mask))
    print(repr(y_test))
    print(repr(test_mask))
