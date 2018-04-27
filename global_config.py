

class Config:
    is_training = True
    batch_norm_epsilon = 1e-5
    size = 256
    batch_norm_decay = 0.9997
    number_of_classes = 5
    l2_regularizer = 0.01
    starting_learning_rate = 0.00001
    multi_grid = [1,2,4]
    output_stride = 16
    resnet_model = "resnet_v2_50"
    batch_size = 16


