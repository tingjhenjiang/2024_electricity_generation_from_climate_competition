# %%
from typing import List, Dict
import pandas as pd
import pytz
import get_data
import numpy as np
import importlib
import pathlib
import json
from IPython.display import clear_output
import pypots.imputation
from pypots.utils.metrics import calc_mae
final_numpy_data_dtype = get_data.final_numpy_data_dtype #'float32'
clear_output()
importlib.reload(get_data)
# %%
if __name__ == '__main__':
    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--modelpath', type=str, default="E:\\scripts\\python\\electricity_yield_prediction_competition\\models\\2bac7eb17c96a8874f9cf5006d1beb1c0b129892d3f1d445d45db8736433aaec\\20241127_T194404\\SAITS_epoch34_loss0.0470.pypots", help="modelpath")
    
    args = parser.parse_args()
    argp_dict = vars(args)
    # argp_dict = {'modelpath':"E:\\scripts\\python\\electricity_yield_prediction_competition\\models\\40b5c1c16e40da557b71efbc4e868582ed81bb5b456ddf5f6cb814d3b14bbdba\\20241128_T114730\\SAITS_epoch6_loss0.0408.pypots"}
    argp_dict['modelpath'] = pathlib.Path(argp_dict['modelpath'])
    hyperparameters = (argp_dict['modelpath'].parent.parent) / "hyperparameter.json"
    hyperparameters = hyperparameters.read_text(encoding='utf-8')
    hyperparameters = json.loads(hyperparameters)

    if (hyperparameters['modeltype']=='SAITS'):
        model = pypots.imputation.saits.SAITS(
            n_steps=hyperparameters['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            d_ffn=128,
            n_heads=4,
            d_k=64,
            d_v=64,
            dropout=0.1,
            epochs=10,
            saving_path=argp_dict['modelpath'].parent, # set the path for saving tensorboard logging file and model checkpoint
            model_saving_strategy="better", # only save the model with the best validation performance
        )
    
    if hyperparameters['modeltype']=='Crossformer':
        model = pypots.imputation.crossformer.Crossformer(
            n_steps=hyperparameters['windowsize'],
            n_features=len(get_data.train_cols),
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_ffn=128,
            factor=5,
            seg_len=8,
            win_size=3,
            batch_size=2, #hyperparameters['batch_size'],
            epochs=hyperparameters['epoch'],
            num_workers=get_data.num_cores,
            saving_path=argp_dict['modelpath'].parent,
            model_saving_strategy="better"
        )

        model.load(str(argp_dict['modelpath']))
    # %%

    alldatadf = get_data.complete_read_dfs(
        comp_folder_name="both_train_comp",
        set_index=True,
        directlyfromfile=False,
        exclude_na_range=False)
    alldatadf = alldatadf.sort_values(by=['LocationCode','DataTime'], ascending=[True, True]).reset_index(drop=True)
    rest_columns = [col for col in alldatadf.columns if col not in get_data.train_cols]
    # 	DataTime	LocationCode	DateTime	StationID	PeakGust.MaximumTime	Precipitation.MeltFlag	StationPressure.Instantaneous	AirTemperature.Instantaneous	RelativeHumidity.Instantaneous	WindSpeed.Mean	WindDirection.Mean	PeakGust.Maximum	PeakGust.Direction	Precipitation.Accumulation	WindSpeed.TenMinutelyMaximum	WindDirection.TenMinutelyMaximum	aq_WD_HR	aq_AMB_TEMP	aq_CH4	aq_CO	aq_NMHC	aq_NO	aq_NO2	aq_NOx	aq_O3	aq_PM10	aq_PM2.5	aq_RAINFALL	aq_RAIN_COND	aq_RH	aq_SO2	aq_THC	aq_WIND_DIREC	aq_WIND_SPEED	aq_WS_HR	StationID_C0Z100	LocationCode_2	LocationCode_3	LocationCode_4	LocationCode_5	LocationCode_6	LocationCode_7	LocationCode_8	LocationCode_9	LocationCode_10	LocationCode_11	LocationCode_12	LocationCode_13	LocationCode_14	LocationCode_15	LocationCode_16	LocationCode_17	basis_datetime	WindSpeed(m/s)	Pressure(hpa)	Temperature(°C)	Humidity(%)	Power(mW)	Sunlight(Lux)	latitude	longitude	LuminationFactor
    comp_pred_target = get_data.custom_read_dfs("36_TestSet_SubmissionTemplate", compute=True)
    comp_pred_target = comp_pred_target.drop(columns=['WindSpeed(m/s)','Pressure(hpa)','Temperature(°C)','Humidity(%)','Sunlight(Lux)','Power(mW)'])

    # comp_pred_target[comp_pred_target['DataTime']<pd.Timestamp('2024-01-17', tz='Asia/Taipei')]

    alldatadf_with_target_indicator = alldatadf.merge(comp_pred_target, how='outer', indicator=True).sort_values(by=['LocationCode','DataTime'], ascending=[True,True]).reset_index(drop=True)
    compute_target_indices = alldatadf_with_target_indicator[alldatadf_with_target_indicator['_merge'].isin(['right_only','both'])].index

    # 	Power(mW)	DataTime	LocationCode	StationID	DateTime
    # %%
    def preprocess_df_to_model_input_dict(srcdf:pd.DataFrame)->np.ndarray:
        targetdf = srcdf.loc[:,get_data.train_cols]
        targetdf = targetdf.astype(final_numpy_data_dtype)
        targetdf = targetdf.to_numpy()
        targetdf = np.expand_dims(targetdf, axis=0)
        targetdf = {'X':targetdf.astype(final_numpy_data_dtype)}
        return targetdf

    def get_sharded_compute_target_indices(srcseries:pd.core.series.Series)->List[List[int]]:
        compute_target_indices_aslist = srcseries.tolist()
        compute_target_index_previous = None
        compute_target_indices_sharded = []
        temp_indices = []
        while True:
            if len(compute_target_indices_aslist)<=0:
                if len(temp_indices)>0:
                    compute_target_indices_sharded.append(temp_indices)
                break
            current_target_index = None
            if compute_target_index_previous is None:
                compute_target_index_previous = compute_target_indices_aslist.pop(0)
                temp_indices.append(compute_target_index_previous)
                is_continuous_with_previous = False
            else:
                current_target_index = compute_target_indices_aslist.pop(0)
                is_continuous_with_previous = (compute_target_index_previous+1)==current_target_index
                if is_continuous_with_previous:
                    temp_indices.append(current_target_index)
                    compute_target_index_previous = current_target_index
                else:
                    compute_target_indices_sharded.append(temp_indices)
                    temp_indices = [current_target_index]
                    compute_target_index_previous = current_target_index
        return compute_target_indices_sharded

    def get_windowsized_sharded_compute_target_indices(
            srclist:List[List[int]],
            windowsize:int=hyperparameters['windowsize']
        )->Dict[str, np.ndarray]:
        get_prediction_blocks = []
        windowsized_sharded_indices_mapping = []
        boolx = []
        for iteri,sharded_compute_target_indices in enumerate(srclist):
            len_sharded_index = len(sharded_compute_target_indices)
            max_index = max(sharded_compute_target_indices)
            q = len_sharded_index//windowsize
            if len_sharded_index % windowsize !=0:
                min_index = max_index-(q+1)*windowsize+1
                q = q+1
            else:
                min_index = min(sharded_compute_target_indices)
                q = q
            tempx = np.arange(min_index, max_index+1, dtype=np.uint)
            tempx_splitted = np.split(tempx, range(windowsize, len(tempx), windowsize))
            tempboolx = [np.isin(x, sharded_compute_target_indices) for x in tempx_splitted]
            get_prediction_blocks.extend(tempx_splitted)
            windowsized_sharded_indices_mapping.extend( [iteri]*q )
            boolx.extend(tempboolx)
        return {
            'get_prediction_blocks':get_prediction_blocks,
            'windowsized_sharded_indices_mapping':windowsized_sharded_indices_mapping,
            'bool_indicator':boolx
        }

    sharded_compute_target_indices = get_sharded_compute_target_indices(compute_target_indices)
    windowsized_sharded_compute_target_indices = get_windowsized_sharded_compute_target_indices(sharded_compute_target_indices)
    # len(windowsized_sharded_compute_target_indices['get_prediction_blocks'])
    # %%
    impute_results = []
    for iteri, compute_target_indices in enumerate(windowsized_sharded_compute_target_indices['get_prediction_blocks']):
        subsetted_locationdf = alldatadf_with_target_indicator.loc[compute_target_indices]
        bool_indicator = windowsized_sharded_compute_target_indices['bool_indicator'][iteri]
        subsetted_location_nparray = preprocess_df_to_model_input_dict(subsetted_locationdf)
        impute_result = model.impute(subsetted_location_nparray)
        impute_result = np.squeeze(impute_result)
        impute_result = pd.DataFrame(data=impute_result, columns=get_data.train_cols)
        impute_result = get_data.sklearn_transform_cols(impute_result, targetcol='competition', inverse=True)
        impute_result.index = subsetted_locationdf.index
        impute_result = subsetted_locationdf.drop(columns=impute_result.columns).join(impute_result)
        impute_result = impute_result[impute_result['_merge'].isin(['both','right_only'])]
        impute_results.append(impute_result)
        # break
    impute_results = pd.concat(impute_results, axis=0)

    # %%
    comp_result = impute_results.loc[:,['DataTime','LocationCode','Power(mW)']]
    comp_result['序號'] = comp_result['DataTime'].dt.strftime('%Y%m%d%H%M')
    comp_result['序號'] = comp_result['序號']+comp_result['LocationCode'].astype(str).str.zfill(2)
    abnormal_negative_predictions = comp_result['Power(mW)']<0
    comp_result.loc[abnormal_negative_predictions,'DataTime']
    comp_result.loc[abnormal_negative_predictions,'Power(mW)'] = 0
    comp_result = comp_result.drop(columns=['DataTime','LocationCode']).rename(columns={'Power(mW)':'答案'})
    comp_result = comp_result.loc[:,['序號','答案']]
    comp_result['序號'] = comp_result['序號'].astype('int64')
    comp_result = get_data.custom_read_dfs("36_TestSet_SubmissionTemplate", preserve_comp_format=True).loc[:,['序號']].merge(comp_result, how='left')
    comp_result.to_csv(get_data.current_dir/"submit.csv", index=False)
    # 20240106090001
    # 20240117091001
# %%