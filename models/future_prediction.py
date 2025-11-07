"""  
Script to predict future time steps using a trained long-term forecasting model.

This script sets up the model parameters, loads the best model checkpoint, and predicts a specified number 
of future time steps based on the last available data point.

Job-SDF original paper doesn't include future prediction, so this is an extension for practical use cases.

Original paper: https://arxiv.org/abs/2406.11920 

@author: Joshua Arrazola
@supervisor: PhD. Sabur Butt
"""

""" 
IMPORTANT: Multivariate models use a sliding window approach for future predictions.

For example, if the model predicts 3 steps ahead but we want to predict 10 steps ahead,
we predict in chunks of 3 steps, updating the input with each prediction until we reach 10 steps.

ex: [1, 2, 3, 4, 5, 6] -> predict [7, 8, 9] -> new input [4, 5, 6, 7, 8, 9] -> predict [10, 11, 12] -> ... and so on.

Also, transformed-based attention models require both encoder and decoder inputs. 
"""


import torch
import os
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from argparse import Namespace

# Set dataset sizes for the different datasets you wanna use
dataset_size = {
    "r0": 2335,
    "region": 16345,
    "r1": 32690,
    "r2": 121420,
    "r1-region": 228830,
    "r2-region": 849940,
    "company": 1216535,
}


# Reconfigure args as needed
args = Namespace(
    # Basic config
    task_name='long_term_forecast',
    is_training=0, # we arent training here
    model_id='0',
    model='LSTM', # <--- important to change based on model used during training

    # Data config
    data='job_demand_region',
    root_path='../../dataset/demand/', # <-- change the path to your dataset
    data_path='r1.parquet', # <-- change based on dataset used during training
    features='M', # <-- time series type (M == multivariate)
    target='OT', 
    freq='m', # frequency of the data (m == monthly)
    checkpoints='.cache/checkpoints', # <-- path to save/load model checkpoints

    # Forecasting
    seq_len=6, # <-- length of input sequence
    label_len=1, # <-- length of label (for attention models)
    pred_len=3, # <-- length of prediction per step
    seasonal_patterns='Monthly',
    inverse=True,

    # Model params
    expand=2,
    d_conv=4,
    top_k=5,
    num_kernels=6,

    # Ensure these match the dataset used during training
    enc_in=dataset_size["r1"],  
    dec_in=dataset_size["r1"],
    c_out=dataset_size["r1"],
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

    # Optimization
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

    # GPU
    use_gpu=False,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1',

    # De-stationary projector
    p_hidden_dims=[128, 128],
    p_hidden_layers=2,

    # Misc
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

args.enc_in = args.dec_in = args.c_out = dataset_size["r1"]

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
        best_model_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        print(f"Loading best model from {best_model_path}")
        exp.model.load_state_dict(torch.load(best_model_path, map_location=device))

    exp.model.eval() # set model to evaluation mode

    test_data, test_loader = exp._get_data(flag='test') # get the test data
    batch_x, batch_y, batch_x_mark, batch_y_mark = list(test_loader)[-1] # get the last batch from test data

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


            # Prepare decoder input
            dec_inp = torch.zeros_like(current_x[:, -args.label_len:, :]).float() 

            # Append zeros for the steps we want to predict
            dec_inp = torch.cat(
                [current_x[:, -args.label_len:, :],
                 torch.zeros((1, steps_to_predict, current_x.shape[-1])).to(device)],
                dim=1
            )

            """  
            If the model uses attention then it returns a 2-tuple (outputs, attns).
            Otherwise, it just returns the outputs.
            """
            if args.output_attention:
                outputs = exp.model(current_x, current_x_mark, dec_inp, current_x_mark)[0]
            else:
                outputs = exp.model(current_x, current_x_mark, dec_inp, current_x_mark)

            # Get the predicted values for the steps we wanted to predict
            pred = outputs[:, -steps_to_predict:, :]

            preds_total.append(pred.cpu()) # store predictions

            # Update current_x with the new predictions for the next iteration
            current_x = torch.cat([current_x, pred], dim=1) 
            current_x_mark = torch.cat([current_x_mark, current_x_mark[:, -steps_to_predict:, :]], dim=1)

            # Decrease the remaining steps
            remaining_steps -= steps_to_predict

    # Concatenate all predictions
    preds_total = torch.cat(preds_total, dim=1)

    # Inverse transform if scaling was applied (idk if this is needed tbh but some models use it so i'll keep it)
    if test_data.scale and args.inverse:
        preds_total = torch.tensor(test_data.inverse_transform(preds_total.numpy())).to(device)

    print(f"Predicted {future_steps} future steps -> shape: {preds_total.shape}")
    return preds_total

if __name__ == "__main__":
    exp = Exp_Long_Term_Forecast(args)

    # Define the experiment setting (should match the one used during training)
    setting = 'long_term_forecast_0_LSTM_job_demand_region_ftM_sl6_ll1_pl3_dm512_nh8_el2_dl2_df2048_expand2_dc4_fc1_eblearned_dtTrue_test_0'
    future_pred = predict_future(exp, setting, future_steps=3, load_best=True)

    print("Predicted future values:")
    print(future_pred) # shape: [1, future_steps, num_features]