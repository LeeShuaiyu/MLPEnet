# config.py
import os

class Config:
    def __init__(self):
        self.data_root = './data'
        self.system_name = 'mlpenet'
        self.save_path = os.path.join(self.data_root, self.system_name)

        param_dic = {
            'eta': [0.0, 5.0],
            'epsilon': [0.0, 0.05],
            'alpha': [1.01, 2.0],
            #'eta': [0, 5.2],
            #'epsilon': [0, 0.06],
            #'alpha': [1.01, 2],
            'param_name': ['eta'],
            'param_name_latex': ['$\\eta$'],
            #'param_name': [ 'epsilon'],
            #'param_name_latex': [ '$\\epsilon$'],
            #'param_name': ['alpha'],
            #'param_name_latex': ['$\\alpha$'],



            'N': [1000, 1000],
            'T': [5, 15],
            'train_num': 5000,
            'eval_num': 2000,
            'test_num': 2000,
            'train_file': os.path.join(self.save_path, 'ou_03_train.pkl'),
            'eval_file': os.path.join(self.save_path, 'ou_03_eval.pkl'),
            'test_file': os.path.join(self.save_path, 'ou_03_test.pkl'),
            'num_epochs': 800,
            'learning_rate': 0.001,
            'batch_size': 800,
            'mlpenet_in_dim': 1,
            'mlpenet_hidden_dim': 50,
            'mlpenet_n_layer': 2,
            'mlpenet_conv_out_channels': 50,
            'mlpenet_conv_kernel_size': 3,
            'mlpenet_activation': 'leakyrelu',
            'mlpenet_init_weight_file': '',
            #'mlpenet_init_weight_file': 'data/mlpenet/mlpenet_model_1/model_mlpenet_model_1_epoch_00120.ckpt',
            'mlpenet_architecture_name': 'mlpenet_ou',
            'mlpenet_drop_last': True,
            'optimizer': 'sam',  # 'adam' or 'sam'
        }
        self.param = param_dic
