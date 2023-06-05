import argparse
import numpy as np
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.1)
    parser.add_argument('--batch_size',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--checkpoint_path',type=str,default=None)
    parser.add_argument('--train_dir',type=str,default=r'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_train.txt')
    parser.add_argument('--test_dir',type=str,default=r'data\WFLW_annotations\WFLW_annotations\list_98pt_rect_attr_train_test\list_98pt_rect_attr_test.txt')
    parser.add_argument('--model_save_path',type=str,default=r'./models')
    parser.add_argument('--log_dir',type=str,default=r'./logs')
    parser.add_argument('--k',type=int,default=1)
    parser.add_argument('--scheduler_step',type=int,default=10)
    return parser.parse_args()