import numpy as np

# load in the data with this function
def load_dataset():
    NUM_SAMPLES = 2959
    DIM = 80
    variables = []
    # variables = np.zeros(shape=(NUM_SAMPLES, DIM))
    target = []
    with open('sfo_data_clean.csv', 'r') as data:
        for idx, row in enumerate(data):
            if idx > 0:
                r = map(int, row.strip().split(','))
                variables.append(r[:-1])
                target.append(r[-1])

    percent_train = 0.8
    last_train = int(percent_train*NUM_SAMPLES)
    variables = np.asarray(variables, dtype=np.float64)
    target = np.asarray(target, dtype=np.int32)
    np.save('train_data', variables[:last_train,:])
    np.save('test_data', variables[last_train:,:])
    np.save('train_targets', target[:last_train])
    np.save('test_targets', target[last_train:])
