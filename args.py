import argparse

parser = argparse.ArgumentParser()
envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument("--size", type=int, default=256, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=5, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.01, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.001, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")
envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")
trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=16, help="Batch size for network train.")

args = parser.parse_args()