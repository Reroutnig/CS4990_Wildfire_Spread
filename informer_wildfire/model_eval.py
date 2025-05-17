import torch
from exp.exp_informer import Exp_Informer

import argparse
import os
# import torch

from exp.exp_informer import Exp_Informer
import numpy as np


parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, required=True, default='custom', help='data')
parser.add_argument('--root_path', type=str, default='./fireData/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='onelasttime.csv', help='data file')    
# parser.add_argument('--data_path', type=str, default='quarter_big_fire_data.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='min_lon', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=16, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=8, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=512, help='output size')

# parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
# parser.add_argument('--dec_in', type=int, default=6, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=6, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'custom':{'data':'onelasttime.csv','T':'min_lon','M':[1,1,1],'S':[1,1,1],'MS':[1,1,1]},
    # 'custom':{'data':'quarter_big_fire_data.csv','T':'min_lon','M':[6,6,6],'S':[1,1,1],'MS':[6,6,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer


import os

args.output_attention = True

exp = Exp(args)

model = exp.model

setting = 'goshDarnIt'
path = os.path.join(args.checkpoints,setting,'checkpoint.pth')
model.load_state_dict(torch.load(path))
print('before?')
batch_y_HOLD = []
def predict(exp, setting, load=False):
    global batch_y_HOLD
    pred_data, pred_loader = exp._get_data(flag='pred')
    print('from this?')
        
    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path+'/'+'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

    exp.model.eval()
        
    preds = []
        
    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # decoder input
        if exp.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len,batch_y.shape[-2], batch_y.shape[-1]]).float()
        elif exp.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len,batch_y.shape[-2], batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:exp.args.label_len,:, :], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast("cuda"):
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features=='MS' else 0
        batch_y = batch_y[:,-exp.args.pred_len:,f_dim:].to(exp.device)
        print(f"batch y? {batch_y.shape}")
        batch_y_HOLD = batch_x
        pred = outputs.detach().cpu().numpy()#.squeeze()
        
        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    
    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # preds = pred_loader.dataset.inverse_transform(preds)
    np.save(folder_path+'real_prediction.npy', preds)
    print(f"{type(pred_loader)}")
    print(f"{type(preds)}")
    # trues = np.load('./results/'+setting+'/true.npy')
    # trues = pred_loader.dataset.inverse_transform(trues)
    
    return preds

prediction = predict(exp, setting, True)
print(prediction.shape)
# print(str(prediction[0]))

#
# path = 'tensorData/'
# pt_files = sorted([f for f in os.listdir(path) if f.endswith('.pt')])
# tensor_list = []
# num = 1
# for file in pt_files:
#     if num == 364:
#         break
#     else:
#         # tensor_path = os.path.join(path, file)
#         tensor = torch.load(os.path.join(path + 'burn_2024_' + str(num) + '.pt'))  # Expecting shape (340, 220)
#         print(os.path.join(path + 'burn_2024_' + str(num) + '.pt'))

#         if tensor.shape != (340, 220):
#             raise ValueError(f"Unexpected shape {tensor.shape} in {file}")

#         tensor_list.append(tensor)
#     num +=1 

# import numpy as np
# import matplotlib.pyplot as plt

# # Create example 2D matrices
# # matrices = [np.random.rand(340, 220) for _ in range(20)]

# index = 0  # Start from the first image
# squeeze = np.squeeze(batch_y_HOLD)
# def show_image(idx):
#     plt.imshow(squeeze[idx], cmap='gray')
#     plt.title(f"Image {idx}")
#     plt.axis('off')
#     plt.draw()

# def on_key(event):
#     global index
#     if event.key == 'right':
#         index = (index + 1) % len(squeeze)
#     elif event.key == 'left':
#         index = (index - 1) % len(squeeze)
#     plt.clf()
#     show_image(index)

# # Start plot
# fig = plt.figure()
# fig.canvas.mpl_connect('key_press_event', on_key)
# show_image(index)
# plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Example 2D matrix (replace this with your actual matrix)
# matrix = np.random.rand(340, 220)  # Values between 0 and 1

# Display as an image
plt.imshow(prediction[1], interpolation='nearest')
plt.colorbar(label='Intensity')  # Optional: shows a color scale
plt.title("Pixel Intensity Grid")
plt.show()



# fig, axes = plt.subplots(2, 2, figsize=(8, 8))  # 2 rows, 2 columns

# titles = ['Matrix 1', 'Matrix 2', 'Matrix 3', 'Matrix 4']

# for ax, mat, title in zip(axes.flatten(), prediction[0:4], titles):
#     ax.imshow(mat, cmap='viridis')
#     ax.set_title(title)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()


# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# # Plot the first matrix
# axes[0].imshow(prediction[0], cmap='viridis')
# axes[0].set_title('Matrix 1')
# axes[0].axis('off')  # Hide axis ticks

# squeeze = np.squeeze(batch_y_HOLD)
# print(f"batch y? {batch_y_HOLD.shape}")

# # Plot the second matrix
# axes[1].imshow(squeeze[0], cmap='viridis')
# axes[1].set_title('Matrix 2')
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()

