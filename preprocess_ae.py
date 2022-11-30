import numpy as np, torch, random
from sklearn.model_selection import train_test_split



dataset = 'AE'
seq_len = 144
min_len, max_len = 5, 20 # retain a percentage between 5% to 20% of the values for a given feature in a data point, so there is 80% to 95% missingness in the feature

num_samples_list = [1, 2, 4, 8, 16, 32]
num_versions = 5
read_path = '/home/ranak/tarnet_data'



def mean_standardize(data, mask):

    features = [[] for i in range(data[0].shape[1])]
    for i in range(len(data)):
        # print(np.sum(mask[i], axis = 0))

        l = []
        for j in range(data[i].shape[1]):
            values = data[i][ : , j][mask[i][ : , j].astype(bool)]
            features[j].extend(values)
            l.append(len(values))
        # print(l)
    
    mean, std = [], []
    for feat in features:
        # print(len(feat))
        feat = np.array(feat)
        mean.append(np.mean(feat))
        std.append(np.std(feat))
    mean, std = np.array(mean), np.array(std)

    for i in range(len(data)): 
        data[i] = (data[i] - mean) / std
        data[i] = np.where(mask[i] == 1, data[i], 0.0)

    return data



def transform_into_irregular_asynchronous_data(dataset):
    time = np.linspace(0, 1, num = dataset.shape[1])
    mask = np.zeros_like(dataset)

    # filter and generate the irregular length data
    new_data, new_mask, new_time = [], [], []
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[2]):
            d, m = dataset[i, : , j], mask[i, : , j]
            num_selected_indices = int((random.randint(min_len, max_len) / 100) * dataset.shape[1])
            selected_indices = random.sample(range(dataset.shape[1]), num_selected_indices)
            m[selected_indices] = 1

            # retain a percentage between 5% to 20% of the values for a given feature in a data point
            sampled_d = np.where(m == 1, d, 0.0) 
            dataset[i, : , j], mask[i, : , j] = sampled_d, m
            # print(mask[i, : , j])
            # print(dataset[i, : , j])
        # print()

        # for times in the data where no feature was sampled (mask sums to 0 for that row), remove that row from the data, mask and time
        new_d = np.delete(dataset[i], np.sum(mask[i], axis = 1) == 0, 0)
        new_m = np.delete(mask[i], np.sum(mask[i], axis = 1) == 0, 0)
        new_t = np.delete(time, np.sum(mask[i], axis = 1) == 0)
        
        # print(new_m)
        # print(new_d)
        # print(new_t)
    
        new_data.append(new_d)
        new_mask.append(new_m)
        new_time.append(new_t)

    new_data = mean_standardize(new_data, new_mask)

    max_seq_len = 0
    for data in new_data: 
        max_seq_len = max(max_seq_len, data.shape[0])
    # print(max_seq_len)

    # padding to make all sequences the same length
    new_dataset = []
    for i in range(len(new_data)):
        d = np.concatenate((new_data[i], np.zeros((max_seq_len - new_data[i].shape[0], dataset.shape[2]))), axis = 0) 
        m = np.concatenate((new_mask[i], np.zeros((max_seq_len - new_mask[i].shape[0], dataset.shape[2]))), axis = 0)
        t = np.concatenate((new_time[i], np.zeros((max_seq_len - new_time[i].shape[0]))), axis = 0)
        t = t.reshape(t.shape[0], 1)
        new_dataset.append(np.concatenate((d, m, t), axis = 1))
    return np.array(new_dataset)



def driver(dataset):
    data_path = read_path + '/' + dataset + '/data/'
    X_train, y_train = np.load(data_path + 'X_train.npy'), np.load(data_path + 'y_train.npy')
    X_test, y_test = np.load(data_path + 'X_test.npy'), np.load(data_path + 'y_test.npy')

    # X_train = np.random.rand(1, 10, 8)
    print(X_train.shape, X_test.shape)
    X_train = transform_into_irregular_asynchronous_data(X_train)
    
    '''
    print(X_train.shape)
    i = np.where(X_train[ : , : , : 24] != 0, 1, 0)
    print(np.sum(i))
    print(np.sum(X_train[ : , : , 24 : 48]))
    print(np.sum(i == X_train[ : , : , 24 : 48]))
    '''

    X_test = transform_into_irregular_asynchronous_data(X_test)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
    print(dataset, X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    train_path = '/home/ranak/iasmts_data/' + dataset + '/regression_data/train/'
    torch.save(torch.tensor(X_train), train_path + 'X_train.pt' )
    torch.save(torch.tensor(y_train), train_path + 'y_train.pt' )
    torch.save(torch.tensor(X_train), '/home/ranak/iasmts_data/' + dataset + '/X_pretrain.pt' )

    val_path = '/home/ranak/iasmts_data/' + dataset + '/regression_data/val/'
    torch.save(torch.tensor(X_val), val_path + 'X_val.pt' )
    torch.save(torch.tensor(y_val), val_path + 'y_val.pt' )
    torch.save(torch.tensor(X_val), '/home/ranak/iasmts_data/' + dataset + '/X_val.pt' )

    test_path = '/home/ranak/iasmts_data/' + dataset + '/regression_data/test/'
    torch.save(torch.tensor(X_test), test_path + 'X_test.pt' )
    torch.save(torch.tensor(y_test), test_path + 'y_test.pt' )

    
    indices = list(range(X_train.shape[0]))
    random.shuffle(indices)
    start, end = 0, 0

    for num_samples in num_samples_list:
        # print('Num Samples: ' + str(num_samples))

        ver = []
        for version in range(num_versions):
            # print('Version: ' + str(version))

            end += num_samples
            if end <= len(indices):
                ver.append(indices[start : end])
            else:
                l = indices[start : end]
                end = end % len(indices)
                l.extend(indices[ : end])
                ver.append(l)
            start = end

        write_path = '/home/ranak/iasmts_data/' + dataset + '/regression_data/' + str(num_samples) + '/'
        for version in range(num_versions):
            X, y = X_train[ver[version]], y_train[ver[version]]
            print(X.shape, y.shape)

            torch.save(torch.tensor(X), write_path + 'X_v' + str(version) + '.pt')
            torch.save(torch.tensor(y), write_path + 'y_v' + str(version) + '.pt')




driver(dataset)