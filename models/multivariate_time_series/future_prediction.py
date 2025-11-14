"""  
Script to predict future time steps using a trained long-term forecasting model.

This script sets up the model parameters, loads the best model checkpoint, and predicts a specified number 
of future time steps based on the last available data point.

Job-SDF original paper doesn't include future prediction, so this is an extension for practical use cases.

Original paper: https://arxiv.org/abs/2406.11920 
Original repository of time series models: https://github.com/thuml/Time-Series-Library

@author: Joshua Arrazola
@supervisor: PhD. Sabur Butt
"""

""" 
IMPORTANT: Multivariate models use a sliding window approach for future predictions.

For example, if the model predicts 3 steps ahead but we want to predict 10 steps ahead,
we predict in chunks of 3 steps, updating the input with each prediction until we reach 10 steps.

ex: [1, 2, 3, 4, 5, 6] -> predict [7, 8, 9] -> new input [4, 5, 6, 7, 8, 9] -> predict [10, 11, 12] -> ... and so on.
"""

"""  
Send as command line arguments:
--seq_len: Input sequence length used during training (default: 6)
--pred_len: Prediction length per step used during training (default: 3)
--model_folder: Folder where model checkpoints are stored (default: 'results/seq_6_len_3')
--model: Model name used during training (default: 'Reformer')
--future_steps: Number of future steps to predict (months [because of montly argument]) (default: 3)
"""

import torch
import os
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace
import numpy as np
import argparse

# Set dataset sizes for the different datasets you wanna use
dataset_size = {
    "green_skills": 274
}

parser = argparse.ArgumentParser()

parser.add_argument('--seq_len', type=int, default=6, help='Input sequence length')
parser.add_argument('--pred_len', type=int, default=3, help='Prediction length per step')
parser.add_argument('--model_folder', type=str, default='results/seq_6_len_3', help='Folder where model checkpoints are stored')
parser.add_argument('--model', type=str, default='Reformer', help='Model name')
parser.add_argument('--future_steps', type=int, default=3, help='Number of future steps to predict')

args_parsed = parser.parse_args()

args = Namespace(
    task_name='long_term_forecast',
    is_training=0,
    model_id='0',
    model=args_parsed.model,           

    data='job_demand_region',
    root_path='../../data/green_skill_classification/',
    data_path='data_for_timeseries.csv',
    features='M',
    target='OT',
    freq='m',
    checkpoints=f'.cache/checkpoints/seq_{args_parsed.seq_len}_len_{args_parsed.pred_len}',

    seq_len=args_parsed.seq_len,       
    label_len=1,
    pred_len=args_parsed.pred_len,
    seasonal_patterns='Monthly',
    inverse=True,

    enc_in=dataset_size["green_skills"],
    dec_in=dataset_size["green_skills"],
    c_out=dataset_size["green_skills"],

    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=2,
    d_ff=2048,
    moving_avg=25,
    factor=1,
    distil=True,
    dropout=0.1,
    embed='learned',
    activation='gelu',
    output_attention=False,
    channel_independence=1,
    decomp_method='moving_avg',
    use_norm=1,
    down_sampling_layers=0,
    down_sampling_window=1,
    down_sampling_method=None,
    seg_len=3,

    num_workers=0,
    itr=1,
    train_epochs=20,
    batch_size=1,
    patience=3,
    learning_rate=0.0001,
    des='test',
    loss='MSE',
    lradj='type1',
    use_amp=False,

    use_gpu=False,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1',

    p_hidden_dims=[128, 128],
    p_hidden_layers=2,

    use_dtw=False,
    augmentation_ratio=0,
    seed=2,
    jitter=False,
    scaling=False,
    permutation=False,
    randompermutation=False,
    magwarp=False,
    timewarp=False,
    windowslice=False,
    windowwarp=False,
    rotation=False,
    spawner=False,
    dtwwarp=False,
    shapedtwwarp=False,
    wdba=False,
    discdtw=False,
    discsdtw=False,
    extra_tag="",
    task_mode="",
    break_mode="exponential",
)

args.enc_in = args.dec_in = args.c_out = dataset_size["green_skills"]

def predict_future(exp, setting, future_steps=3, load_best=True):
    """ Predict future time steps using the trained model.
    Args:
        exp: Experiment object containing the model and data.
        setting: String identifier for the experiment setting.
        future_steps: Number of future time steps to predict.
        load_best: Whether to load the best model checkpoint.
    Returns:
        preds_total: Tensor containing the predicted future values.
        (shape: [1, future_steps, num_features])
    """

    args = exp.args # get args from experiment
    device = exp.device # get device from experiment

    # Load the best model checkpoint
    if load_best:
        best_model_path = os.path.join(args.checkpoints, setting, 'best_model.pth') 
        exp.model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        pass 
    
    exp.model.eval() 
    test_data, test_loader = exp._get_data(flag='test') # get the test data

    batch_x, batch_y, batch_x_mark, batch_y_mark = list(test_loader)[-1] 

    print("****")
    print(batch_x, batch_x.shape)
    print(batch_y, batch_y.shape)
    print(batch_x_mark, batch_x_mark.shape)
    print(batch_y_mark, batch_y_mark.shape)
    print("****")

    # batch_x = (batch_size, seq_len, num_features)
    # batch_y = (batch_size, label_len + pred_len, num_features)

    # batch_x_mark = (batch_size, seq_len, num_time_features)
    # batch_y_mark = (batch_size, label_len + pred_len, num_time_features)

    current_x = batch_x[-1:, :, :].float().to(device)
    current_x_mark = batch_x_mark[-1:, :, :].float().to(device)

    preds_total = []
    remaining_steps = future_steps # steps left to predict

    with torch.no_grad(): # turn off gradients (we're only predicting)
        while remaining_steps > 0: # predict until we reach the desired future steps

            """  
            steps left to predict may be less than args.pred_len (model's prediction length),
            so we predict the minimum of the two.

            ex: if we are predicting windows of 3 steps but only have 2 steps left to predict,
            we only predict 2 steps this iteration (i.e. min(3, 2) == 2).
            """
            steps_to_predict = min(args.pred_len, remaining_steps) 

            # Append zeros for the steps we want to predict
            dec_inp = torch.cat(
                [current_x[:, -args.label_len:, :],
                 torch.zeros((1, steps_to_predict, current_x.shape[-1])).to(device)],
                dim=1
            )

            # Create the mask for the decoder input
            dec_inp_mark = torch.cat(
                [current_x_mark[:, -args.label_len:, :],
                current_x_mark[:, -steps_to_predict:, :]],
                dim=1
            )

            """  
            If the model uses attention then it returns a 2-tuple (outputs, attns).
            Otherwise, it just returns the outputs.
            """
            if args.output_attention:
                outputs = exp.model(current_x, current_x_mark, dec_inp, dec_inp_mark)[0]
            else:
                outputs = exp.model(current_x, current_x_mark, dec_inp, dec_inp_mark)

            pred = outputs[:, -steps_to_predict:, :]
            preds_total.append(pred.cpu()) 

            current_x = torch.cat([current_x, pred], dim=1) 
            current_x_mark = torch.cat([current_x_mark, current_x_mark[:, -steps_to_predict:, :]], dim=1)

            remaining_steps -= steps_to_predict

    preds_total = torch.cat(preds_total, dim=1)

    # Inverse transform if scaling was applied (idk if this is needed tbh but some models use it so i'll keep it)
    if test_data.scale and args.inverse:
        preds_total = torch.tensor(test_data.inverse_transform(preds_total.numpy())).to(device)
        preds_total = torch.clamp(preds_total, min=0)

    print(f"Predicted {future_steps} future steps -> shape: {preds_total.shape}")
    return preds_total

if __name__ == "__main__":
    exp = Exp_Long_Term_Forecast(args)

    setting = args_parsed.model_folder
    future_pred = predict_future(exp, setting, future_steps=args_parsed.future_steps, load_best=True)

    print("Predicted:")
    print(future_pred) # shape: [1, future_steps, num_features]
    torch.save(future_pred, f'../predictions/future_predictions_seq{args_parsed.seq_len}_pred{args_parsed.pred_len}_{args_parsed.model}.pt')