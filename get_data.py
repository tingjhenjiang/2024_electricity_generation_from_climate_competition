# %%
import os, sys
import pandas as pd
import numpy as np
import multiprocessing
import dask, dask.dataframe
import pickle
import sklearn.preprocessing
import pathlib
from typing import Union, Literal, List
import datetime
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from IPython.display import display
import importlib
import torch
import math
import dask.array
import dask.bag

import fetch_cwa_data
import train_lumination

import pvlib
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


importlib.reload(train_lumination)
importlib.reload(fetch_cwa_data)

standard_coordinate = train_lumination.standard_coordinate

pd.options.display.max_columns = None
num_cores = multiprocessing.cpu_count()
dask_compute_kwargs = {'scheduler':'threads','num_workers':num_cores}
timezone = train_lumination.timezone # "Asia/Taipei"
min_time_interval = train_lumination.min_time_interval #'10min'
locationcode_dtype = train_lumination.locationcode_dtype #'int8'#pd.Int8Dtype()
final_numpy_data_dtype = 'float32'
current_folder = pathlib.Path.cwd()


def print_series_type(srcobj_str:str, srcseries:pd.Series):
    print(f"{srcobj_str}: ")
    print(f"srcseries dtype {srcseries.dtype} type {type(srcseries)}")
def completedisplay(df:pd.DataFrame):
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
        display(df)
def rmsle(y_true, y_pred):
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    squared_log_error = np.square(log_true - log_pred)
    mean_squared_log_error = np.mean(squared_log_error)
    rmsle_value = np.sqrt(mean_squared_log_error)
    return rmsle_value
def mspe(y_true, y_pred):
    percentage_errors = ((y_true - y_pred) / y_true) ** 2
    mspe_value = np.mean(percentage_errors) * 100
    return mspe_value
current_file = pathlib.Path(__file__).resolve()
current_dir = current_file.parent
sklearn_encoders = {
    'StationID': "skcoltransformer_dummy_StationID.pkl",
    'LocationCode': "skcoltransformer_dummy_LocationCode.pkl",
    'cwa': "skcoltransformer_min_max_allcwa.pkl",
    'competition': "skcoltransformer_min_max_allcompetition.pkl",
    'airquality': "skcoltransformer_min_max_airquality.pkl",
}
sklearn_encoders = {k:current_dir/v for k,v in sklearn_encoders.items()}
sklearn_encoders = {k:pickle.load(open(v,'rb')) if sklearn_encoders[k].is_file() else sklearn_encoders[k] for k,v in sklearn_encoders.items()}
train_cols:List=[
        'Precipitation.MeltFlag','Precipitation.Accumulation',
        'StationPressure.Instantaneous','AirTemperature.Instantaneous',
        'RelativeHumidity.Instantaneous','WindSpeed.Mean',
        # 'WindDirection.Mean',
        # 'PeakGust.Maximum','PeakGust.Direction',
        # 'WindSpeed.TenMinutelyMaximum','WindDirection.TenMinutelyMaximum',
        'GlobalSolarRadiation.Accumulation',
        'Visibility.Instantaneous',
        'TotalCloudAmount.Instantaneous',
        'aq_AMB_TEMP',
        # 'aq_CH4','aq_CO','aq_NMHC','aq_NO','aq_NO2',
        # 'aq_NOx','aq_O3',
        'aq_PM10','aq_PM2.5','aq_RAINFALL','aq_RAIN_COND',
        # 'aq_RH','aq_SO2','aq_THC','aq_WIND_DIREC','aq_WIND_SPEED',
        # 'aq_WS_HR',
        'StationID_C0Z100','LocationCode_2','LocationCode_3','LocationCode_4','LocationCode_5','LocationCode_6',
        'LocationCode_7','LocationCode_8','LocationCode_9','LocationCode_10','LocationCode_11','LocationCode_12',
        'LocationCode_13','LocationCode_14','LocationCode_15','LocationCode_16','LocationCode_17','WindSpeed(m/s)',
        'Pressure(hpa)','Temperature(°C)','Humidity(%)','Power(mW)','Sunlight(Lux)',
        # 'latitude','longitude',
        'LuminationFactor','effective_irradiance']

location_cwastation_pairs = {
    17:'466990',	
    16:'466990',
    15:'466990',
    14:'C0Z100',
    13:'C0Z100',
    12:'C0Z100',
    11:'C0Z100',
    10:'C0Z100',
    9:'C0Z100',
    8:'C0Z100',
    7:'C0Z100',
    6:'C0Z100',
    5:'C0Z100',
    4:'C0Z100',
    3:'C0Z100',
    2:'C0Z100',
    1:'C0Z100',
}
location_cwastation_pairs = pd.DataFrame.from_dict(location_cwastation_pairs, orient='index').reset_index().rename(columns={'index':'LocationCode',0:'StationID'})
location_cwastation_pairs['LocationCode'] = location_cwastation_pairs['LocationCode'].astype(locationcode_dtype)
location_cwastation_pairs_dd = dask.dataframe.from_pandas(location_cwastation_pairs, npartitions=1)


def sklearn_transform_cols(
        srcdf:pd.DataFrame,
        targetcol:Literal['StationID','LocationCode','cwa','competition','airquality'],#Union[List[str], Literal['StationID','LocationCode','lumination']]
        inverse:bool=False
    )->pd.DataFrame:
    if len(srcdf.index)==0:
        return srcdf
    transform_cols = {
        'StationID':[targetcol],
        'LocationCode':[targetcol],
        'cwa':['StationPressure.Instantaneous', 'AirTemperature.Instantaneous', 'RelativeHumidity.Instantaneous',
            'WindSpeed.Mean',
            'GlobalSolarRadiation.Accumulation',
            'Visibility.Instantaneous',
            'TotalCloudAmount.Instantaneous',
            # 'WindDirection.Mean', 'PeakGust.Maximum',
            # 'PeakGust.Direction', 'Precipitation.Accumulation', 'WindSpeed.TenMinutelyMaximum', 'WindDirection.TenMinutelyMaximum',
            ],
        'competition':['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Power(mW)', 'Sunlight(Lux)', 'LuminationFactor', 'effective_irradiance'],
        'airquality':['aq_AMB_TEMP', 'aq_CH4', 'aq_CO', 'aq_NMHC', 'aq_NO', 'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM10', 'aq_PM2.5', 'aq_RAINFALL', 'aq_RAIN_COND', 'aq_RH', 'aq_SO2', 'aq_THC', 'aq_WIND_DIREC', 'aq_WIND_SPEED', 'aq_WS_HR'],
    }[targetcol]
    if isinstance(sklearn_encoders[targetcol],pathlib.Path):
        onehotencdtype = locationcode_dtype if targetcol=='LocationCode' else str
        enc = sklearn.preprocessing.OneHotEncoder(
            drop='first',handle_unknown='ignore',sparse_output=False, dtype=onehotencdtype
            ) if targetcol in ['StationID','LocationCode'] else sklearn.preprocessing.MinMaxScaler()
        enc.fit(srcdf[transform_cols])
        with open(sklearn_encoders[targetcol], 'wb') as f:
            pickle.dump(enc, f)
    else:
        enc = sklearn_encoders[targetcol]
    # print(f"targetcol is {targetcol} transform_cols is {transform_cols}")
    transformed_df_columns_names = enc.get_feature_names_out(transform_cols) if targetcol in ['StationID','LocationCode'] else transform_cols
    if not inverse:
        transformed_data = enc.transform(srcdf[transform_cols])
    else:
        transformed_data = enc.inverse_transform(srcdf[transform_cols])
    transformed_df = pd.DataFrame(
        data=transformed_data,
        columns=transformed_df_columns_names
    )
    original_df = srcdf.drop(columns=transform_cols) if targetcol not in ['StationID','LocationCode'] else srcdf
    transformed_df = transformed_df.set_index(srcdf.index)
    targetdf = pd.concat([original_df,transformed_df], axis=1)
    return targetdf

def generate_all_range_datatime_df(start:str='2024-01-01 06:30:00', end:str='2024-10-31 18:00:00', locationcode=range(1,18), dftype:Literal['pd','dd']='dd')->Union[pd.DataFrame,dask.dataframe.DataFrame]:
    global locationcode_dtype
    global min_time_interval
    alldata_daterange = pd.date_range(start=start, end=end, freq=min_time_interval, tz=timezone).to_frame().rename(columns={0:'DataTime'})
    alldata_daterange = alldata_daterange.merge(
        pd.Series(locationcode, name='LocationCode').astype(locationcode_dtype).to_frame(),
        how='cross'
    )
    alldata_daterange['LocationCode'] = alldata_daterange['LocationCode'].astype(locationcode_dtype)
    alldata_daterange = alldata_daterange.merge(location_cwastation_pairs, how='left', on=['LocationCode'])
    if dftype=='dd':
        alldata_daterange = dask.dataframe.from_pandas(alldata_daterange, npartitions=1)
    return alldata_daterange

def get_air_pollution_data(join_locations:bool=True, dftype:Literal['pd','dd']='dd')->Union[pd.DataFrame,dask.dataframe.DataFrame]:
    global timezone
    global locationcode_dtype
    from chardet.universaldetector import UniversalDetector
    from io import StringIO
    detector = UniversalDetector()
    detector.reset()
    airquality_src_filepath = current_dir/"airquality.csv"
    for line in open(airquality_src_filepath, 'rb'):
        detector.feed(line)
        if detector.done: break
    filetexts = airquality_src_filepath.read_text(encoding=detector.result['encoding'])
    filetexts = filetexts.splitlines(True)
    filetexts = "".join(filetexts[2:])
    filedata = StringIO(filetexts)
    # Open the file as binary data
    df = pd.read_csv(filedata)
    df['測項'] = df['測項'].astype('category').apply(lambda x: 'aq_'+x)
    df = dask.dataframe.from_pandas(df)
    df = df.melt(id_vars=['測站','日期','測項'],
                 value_vars=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'],
                 var_name='hour',
                 value_name='measured_value')
    df['日期'] = dask.dataframe.to_datetime(df['日期']).dt.tz_localize(timezone)
    timedelta = df['hour'].astype('Int8')
    timedelta = dask.dataframe.to_timedelta(timedelta, unit='hour')
    df['日期'] = df['日期']+timedelta#+pd.to_timedelta(timedelta, unit='hour')
    df = df.drop(columns=['測站','hour'])
    df = df.replace(np.nan, None)
    df = df.replace(['NR'], value='0.0')
    df = df.replace(['*', '#', np.nan, 'x', 'A', '', ' ', None], value=pd.NA)
    df['measured_value'] = df['measured_value'].astype('Float64')
    df = df.pivot_table(index='日期', columns='測項', values='measured_value')
    df = df.reset_index(drop=False).rename(columns={'日期':'DataTime'})
    datatime_series = df['DataTime'].copy()
    df = df.drop(columns='DataTime')
    is_not_all_nan = df.copy()
    for col in is_not_all_nan.columns:
        is_not_all_nan[col] = is_not_all_nan[col].map(lambda x: math.isnan(x) or pd.isnull(x), meta=(col, bool))
    is_not_all_nan = is_not_all_nan.apply(sum,axis=1,meta=('sum',float))
    is_not_all_nan = is_not_all_nan!=df.shape[1]
    df = df[is_not_all_nan]
    df = df.map_partitions(sklearn_transform_cols, targetcol='airquality')
    
    df['DataTime'] = datatime_series
    df.columns.name = None
    if join_locations:
        df_locations = generate_all_range_datatime_df(dftype='dd')
        df_locations = df_locations.drop(columns=['DataTime'])
        df_locations = df_locations.drop_duplicates()
        df_locations = df_locations.assign(joinkey='join')
        df = df.assign(joinkey='join')
        df = df.merge(df_locations, how='left', on='joinkey')
        df = df.drop(columns=['joinkey'])
    if dftype=='pd':
        df = df.compute() #dask.dataframe.from_pandas(df, npartitions=1)
    return df



# %%
def custom_read_dfs(
        type:Literal["sunrisesunset","36_TrainingData","cwa","36_TestSet_SubmissionTemplate","both_train_comp"],
        compute:bool=True,
        preserve_comp_format:bool=False,
    )->Union[pd.DataFrame,dask.dataframe.DataFrame]:
    global timezone
    global min_time_interval
    global locationcode_dtype
    data_folder = current_dir/type
    data_folder_files = data_folder.rglob('*.*')
    data_folder_files = list(data_folder_files)
    global location_cwastation_pairs_dd
    rebuild_sklearn_transformer = False
    if type=="sunrisesunset":
        df = dask.dataframe.from_map(lambda x: pd.read_excel(x, index_col=False, engine="odf"), data_folder_files)
        df = df.rename(columns={"方位":"日出方位","方位.1":"日落方位"})
        df = df.loc[:,[n for n in df.columns if n.find('Unnamed')==-1]] #dropna(axis=1)
        df = df.compute(**dask_compute_kwargs) if compute else df
    elif type=="cwa":
        rebuild_sklearn_transformer = True if isinstance(sklearn_encoders['cwa'],pathlib.Path) else False
        df = dask.dataframe.read_parquet(data_folder)
        if not rebuild_sklearn_transformer:
            df = df.map_partitions(sklearn_transform_cols, targetcol='cwa')
        df = df.merge(location_cwastation_pairs_dd, how='left', on=['StationID'])
        df = df[~df['LocationCode'].isnull()]
        # print("location code null df:")
        # completedisplay(df[df['LocationCode'].isnull()].compute(**dask_compute_kwargs))
        df['LocationCode'] = df['LocationCode'].astype(locationcode_dtype)
        timedelta = pd.to_timedelta(1, unit='min')
        df['DataTime'] = df['DataTime']+timedelta
        df['DataTime'] = df['DataTime'].dt.floor(freq='h')
        # df = df.compute(**dask_compute_kwargs) if compute else df
        if rebuild_sklearn_transformer:
            if not isinstance(df, pd.DataFrame):
                df = df.compute(**dask_compute_kwargs)
            df = sklearn_transform_cols(df, targetcol='cwa')
        if isinstance(df, pd.DataFrame) and not compute:
            df = dask.dataframe.from_pandas(df, npartitions=dask_compute_kwargs['num_workers'])
        if compute:
            df.set_index('DataTime', inplace=True)
            df.sort_index(inplace=True)
    elif type=='36_TestSet_SubmissionTemplate':
        df = pd.read_csv(data_folder_files[0])
        # 西元年(4 碼)+月(2 碼)+日(2 碼)+預測時間(4 碼)+裝置代號 (2 碼)
        df['DataTime'] = df['序號'].apply(lambda x: str(x)[:12])
        df['LocationCode'] = df['序號'].apply(lambda x:f"{str(x)[12:14]}")
        df['LocationCode'] = df['LocationCode'].astype(locationcode_dtype)
        location_cwastation_pairs_pd = location_cwastation_pairs_dd.compute(**dask_compute_kwargs)
        location_cwastation_pairs_pd['LocationCode'] = location_cwastation_pairs_pd['LocationCode'].astype(locationcode_dtype)
        df = df.merge(right=location_cwastation_pairs_pd, how='left', on=['LocationCode'])
        df['DataTime'] = pd.to_datetime(df['DataTime'], format="%Y%m%d%H%M").dt.tz_localize(timezone)
        df['DateTime'] = df['DataTime']
        for col in ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']:
            df[col] = np.nan
        if not preserve_comp_format:
            df = df.drop(columns=['序號']).rename(columns={'答案':'Power(mW)'})
        df = dask.dataframe.from_pandas(df) if not compute else df
    elif type=='36_TrainingData':
        df = dask.dataframe.read_csv(data_folder_files, index_col=False, infer_datetime_format=False)
        df['DateTime'] = df['DateTime'].astype('str').apply(lambda x:x+'000', meta=('DateTime','str'))
        df['DateTime'] = dask.dataframe.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S.%f')
        df['DateTime'] = df['DateTime'].dt.tz_localize(timezone)

        df['LocationCode'] = df['LocationCode'].astype(locationcode_dtype)

        df['DataTime'] = df['DateTime'].dt.floor(freq=min_time_interval)
        df['DateTime'] = (df['DateTime']-pd.Timestamp("1970-01-01", tz=timezone)) // pd.Timedelta('1s')
        df = df.groupby(['DataTime','LocationCode']).mean(numeric_only=True).reset_index(drop=False)
        df['DateTime'] = dask.dataframe.to_datetime(df['DateTime'], unit='s').dt.tz_localize(timezone)

        # df['DateTime'] = df['DateTime'].dt.tz_localize(timezone)
        # df['hourofday'] = df['DateTime'].dt.hour+df['DateTime'].dt.minute/60+df['DateTime'].dt.hour/60/60
        # df['minuteofday'] = df['DateTime'].dt.hour*60+df['DateTime'].dt.minute+df['DateTime'].dt.second/60
        # df['secondofday'] = df['DateTime'].dt.hour*60*60+df['DateTime'].dt.minute*60+df['DateTime'].dt.second
        # df['dayofyear'] = (df['DateTime'].dt.dayofyear).astype('float64')
        df = df[(df['Sunlight(Lux)']<=117758.1) & (df['Sunlight(Lux)']>20.0)]
        no_reason_index = (df['Power(mW)']<=0) & (df['Sunlight(Lux)']>0.0)
        no_reason_index = (no_reason_index) | (df['Pressure(hpa)']<=0.0) | (df['Humidity(%)']<=0.0)
        df = df[~no_reason_index]
        df = df.merge(right=location_cwastation_pairs_dd, how='left', on=['LocationCode'])
        df = df.compute(**dask_compute_kwargs) if compute else df
        # print(df.columns)
        # df = sklearn_transform_cols(df, targetcol='LocationCode')
        if compute:
            # df.set_index(df['DateTime'].dt.floor('min'), inplace=True)
            # df.sort_index(inplace=True)
            pass
    elif type=="both_train_comp":
        df = dask.dataframe.concat([
            custom_read_dfs(type="36_TrainingData", compute=compute),
            custom_read_dfs(type="36_TestSet_SubmissionTemplate", compute=compute)
        ], axis=0)

    return df

# ghi全天空水平面日射量	dni法線面直達日射量	dhi水平面擴散日射量


sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
# sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
# cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']


def calculate_lumination_factor_column(srcdf:Union[pd.DataFrame,dask.dataframe.DataFrame])->pd.Series:
    global train_lumination
    print(f"srcdf.shape = {srcdf.shape}")
    # srcdf.columns = Index(['DataTime', 'LocationCode', 'DateTime', 'WindSpeed(m/s)',
    #        'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)',
    #        'Power(mW)', 'StationID', 'PeakGust.MaximumTime',
    #        'Precipitation.MeltFlag', 'StationPressure.Instantaneous',
    #        'AirTemperature.Instantaneous', 'RelativeHumidity.Instantaneous',
    #        'WindSpeed.Mean', 'WindDirection.Mean', 'PeakGust.Maximum',
    #        'PeakGust.Direction', 'Precipitation.Accumulation',
    #        'WindSpeed.TenMinutelyMaximum', 'WindDirection.TenMinutelyMaximum',
    #        'GlobalSolarRadiation.Accumulation', 'Visibility.Instantaneous',
    #        'TotalCloudAmount.Instantaneous', 'surface_azimuth', 'latitude',
    #        'longitude', 'aq_WD_HR', 'aq_AMB_TEMP', 'aq_CH4', 'aq_CO', 'aq_NMHC',
    #        'aq_NO', 'aq_NO2', 'aq_NOx', 'aq_O3', 'aq_PM10', 'aq_PM2.5',
    #        'aq_RAINFALL', 'aq_RAIN_COND', 'aq_RH', 'aq_SO2', 'aq_THC',
    #        'aq_WIND_DIREC', 'aq_WIND_SPEED', 'aq_WS_HR', 'StationID_C0Z100',
    #        'LocationCode_2', 'LocationCode_3', 'LocationCode_4', 'LocationCode_5',
    #        'LocationCode_6', 'LocationCode_7', 'LocationCode_8', 'LocationCode_9',
    #        'LocationCode_10', 'LocationCode_11', 'LocationCode_12',
    #        'LocationCode_13', 'LocationCode_14', 'LocationCode_15',
    #        'LocationCode_16', 'LocationCode_17'],
    tempdf = srcdf.copy().reset_index(drop=True)
    datetimeindex = pd.DatetimeIndex(tempdf['DataTime'].reset_index(drop=True))
    # print_series_type("srcdf['DataTime']", srcdf['DataTime'])
    # print_series_type("datetimeindex", datetimeindex)
    # print(f"datetimeindex = {datetimeindex}")
    # print("DataTime = ")
    # completedisplay(tempdf['DataTime'])
    # print("datetimeindex =")
    # completedisplay(datetimeindex)
    # tilt_series = pd.Series([20]) #pd.Series([20]*srcdf.shape[0])
    # altitude_series = pd.Series([0]) #pd.Series([0]*srcdf.shape[0])
    full_temperature_series = tempdf['Temperature(°C)']
    full_temperature_series.loc[full_temperature_series.isna()] = tempdf['AirTemperature.Instantaneous'][full_temperature_series.isna()]
    full_temperature_series.loc[full_temperature_series.isna()] = tempdf['aq_AMB_TEMP'][full_temperature_series.isna()]
    full_temperature_series = full_temperature_series.reset_index(drop=True)
    full_pressure_series = tempdf['Pressure(hpa)']
    full_pressure_series.loc[full_pressure_series.isna()] = tempdf['StationPressure.Instantaneous'][full_pressure_series.isna()]
    full_pressure_series = full_pressure_series*100
    full_pressure_series = full_pressure_series.reset_index(drop=True)
    full_windspeed_series = tempdf['WindSpeed(m/s)']
    full_windspeed_series.loc[full_windspeed_series.isna()] = tempdf['WindSpeed.Mean'][full_windspeed_series.isna()]
    full_windspeed_series.loc[full_windspeed_series.isna()] = tempdf['aq_WIND_SPEED'][full_windspeed_series.isna()]
    full_windspeed_series = full_windspeed_series.reset_index(drop=True)
    solpos = pvlib.solarposition.get_solarposition(
        time=datetimeindex,
        latitude=tempdf['latitude'],
        longitude=tempdf['longitude'],
        altitude=tempdf['altitude'],
        temperature=full_temperature_series,
        pressure=full_pressure_series,
    )
    solpos_index = solpos.index
    solpos = solpos.reset_index(drop=True)
    # for col in solpos.columns:
    #     print_series_type(col, solpos[col])
    dni_extra = pvlib.irradiance.get_extra_radiation(datetimeindex) #.dt.dayofyear
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pvlibpressure = pvlib.atmosphere.alt2pres(tempdf['altitude'])
    # print(f"airmass dtype {airmass.dtype} type {type(airmass)}")
    # print(f"airmass = {airmass.sort_values().unique()}")
    # print(f"pvlibpressure dtype {pvlibpressure.dtype} type {type(pvlibpressure)}")
    # print(f"pvlibpressure = {pvlibpressure.sort_values().unique()}")
    # airmass = airmass.astype(pd.Float64Dtype()).reset_index(drop=True)
    # pvlibpressure = pvlibpressure.astype(pd.Float64Dtype()).reset_index(drop=True)
    # am_abs = airmass * pvlibpressure / 101325.0
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pvlibpressure)
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tempdf['surface_tilt'],
        surface_azimuth=tempdf['surface_azimuth'],
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
    )
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tempdf['surface_tilt'],
        surface_azimuth=tempdf['surface_azimuth'],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=tempdf['GlobalSolarRadiation.Accumulation']/0.75,
        ghi=tempdf['GlobalSolarRadiation.Accumulation'],
        dhi=tempdf['GlobalSolarRadiation.Accumulation']-(tempdf['GlobalSolarRadiation.Accumulation']/0.75*np.cos(solpos['apparent_elevation'])),
        dni_extra=dni_extra.reset_index(drop=True),
        airmass=airmass.reset_index(drop=True),
        model='haydavies',
    )
    cell_temperature = pvlib.temperature.sapm_cell(
        total_irradiance['poa_global'],
        full_temperature_series,
        full_windspeed_series,
        **temperature_model_parameters,
    )
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irradiance['poa_direct'],
        total_irradiance['poa_diffuse'],
        am_abs,
        aoi,
        module,
    )
    # dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
    # ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)
    # annual_energy = ac.sum()
    tempdf['effective_irradiance'] = effective_irradiance
    # notna_radiation = (~tempdf['GlobalSolarRadiation.Accumulation'].isna())

    is_notna_datetime = tempdf['DateTime'].notnull()
    tempdf['basis_datetime'] = tempdf['DateTime'].copy()
    tempdf['basis_datetime'].loc[~is_notna_datetime] = tempdf['DataTime'].loc[~is_notna_datetime]
    tempdf['basis_datetime_floored'] = tempdf['basis_datetime'].dt.floor(freq='1s')
    tempdf['LuminationFactor'] = tempdf.apply(
        lambda x: train_lumination.get_lumination_factor(x['basis_datetime_floored'],x['latitude'],x['longitude']),
        axis=1,
    )
    nolights = tempdf['LuminationFactor']<=0
    tempdf.loc[nolights, 'Power(mW)'] = 0
    tempdf.loc[nolights, 'effective_irradiance'] = 0
    tempdf = tempdf.drop(columns=['basis_datetime_floored'])

    return tempdf

def complete_read_dfs(
        compute:bool=True,
        directlyfromfile:bool=True,
        exclude_na_range:bool=True,
        comp_folder_name:Literal["36_TrainingData","36_TestSet_SubmissionTemplate","both_train_comp"]='36_TrainingData',
        set_index:bool=False
    ):
    rebuild_sklearn_transformer = {
        key: True if isinstance(sklearn_encoders[key],pathlib.Path) else False
        for key in sklearn_encoders.keys()
    }
    if directlyfromfile:
        global current_dir
        df = dask.dataframe.read_parquet(current_dir/"alldatadf.parquet")
    else:
        location_coord_azimuth_pairs_dd = dask.dataframe.from_pandas(train_lumination.location_coord_azimuth_pairs_df, npartitions=1)
        location_coord_azimuth_pairs_dd['LocationCode'] = location_coord_azimuth_pairs_dd['LocationCode'].astype(locationcode_dtype)
        df_cwa = custom_read_dfs('cwa', compute=False)
        df_competition = custom_read_dfs(comp_folder_name, compute=False)
        df = df_competition.merge(df_cwa, how='outer', on=['DataTime','LocationCode','StationID'])
        df = df.merge(generate_all_range_datatime_df(), on=['DataTime','LocationCode','StationID'], how='right')
        df = df.merge(location_coord_azimuth_pairs_dd, how='left', on=['LocationCode'])
        df = df.merge(get_air_pollution_data(join_locations=True,dftype='dd'), how='left', on=['DataTime','LocationCode','StationID'])
        # print(f"df.cols {df.columns}")
        if not rebuild_sklearn_transformer['StationID']:
            df = df.map_partitions(sklearn_transform_cols, targetcol='StationID')
        if not rebuild_sklearn_transformer['LocationCode']:
            df = df.map_partitions(sklearn_transform_cols, targetcol='LocationCode') #共有17個水準，轉換為16 columns
        # df = df.compute(**dask_compute_kwargs)
        df = df.map_partitions(calculate_lumination_factor_column)
        # df = calculate_lumination_factor_column(df)
        # df = dask.dataframe.from_pandas(df, npartitions=dask_compute_kwargs['num_workers'])
        if not rebuild_sklearn_transformer['competition']:
            df = df.map_partitions(sklearn_transform_cols, targetcol='competition')
        df = df.drop(columns=['StationID_C0T9E0','StationID_C0T9G0','StationID_C0Z160','StationID_C0Z180','StationID_C0Z200'], errors='ignore')
    if compute:
        df = df.compute(**dask_compute_kwargs)
        df = df.drop_duplicates(subset=['LocationCode', 'basis_datetime'], keep='first')
        if rebuild_sklearn_transformer['StationID']:
            df = sklearn_transform_cols(df, targetcol='StationID')
        if rebuild_sklearn_transformer['LocationCode']:
            df = sklearn_transform_cols(df, targetcol='LocationCode') #共有17個水準，轉換為16 columns
        if rebuild_sklearn_transformer['competition']:
            df = sklearn_transform_cols(df, targetcol='competition')
            pass
        df = df.sort_values(by=['LocationCode', 'DataTime'], ascending=[True, True])
        df = df.reset_index(drop=True)

    if exclude_na_range:
        exclude_missing_conditions = {}
        exclude_missing_condition_final = None
        for locationcode in df['LocationCode'].unique():
            location_subset = df[(df['LocationCode']==locationcode) & (~df['Sunlight(Lux)'].isnull())]
            min_pos = location_subset['DataTime'].min()
            max_pos = location_subset['DataTime'].max()
            exclude_missing_conditions[locationcode] = (df['LocationCode']==locationcode) & (  (df['DataTime']>=min_pos) & (df['DataTime']<=max_pos)   )
            print(f"there are {exclude_missing_conditions[locationcode].sum()} rows for locationcode {locationcode}")
            if exclude_missing_condition_final is None:
                exclude_missing_condition_final = exclude_missing_conditions[locationcode]
            else:
                exclude_missing_condition_final = (exclude_missing_condition_final) | exclude_missing_conditions[locationcode]
        df = df[exclude_missing_condition_final].reset_index(drop=True)
        pass

    if set_index:
        df = df.set_index(df['DataTime'])
        df.index.name = 'DataTime_index'
        df.sort_index(inplace=True)
        pass
    return df
# %%
windowsize = 30
observation_step_gap = 6
def rowindex_to_datadfarray(iteri_this_location_block:Union[int,np.ndarray],
        locationcode:Union[int,np.ndarray]=1,
        windowsize:Union[int,np.ndarray]=windowsize,
        srcdf:pd.DataFrame=pd.DataFrame(),
        train_cols=train_cols,
    ) -> np.ndarray:
    n_features = len(train_cols)
    df_location = srcdf[srcdf['LocationCode']==locationcode].reset_index(drop=True)
    nrows_padnan = windowsize-(df_location.shape[0] % windowsize)
    end_iteri = df_location.shape[0] + nrows_padnan - windowsize
    is_need_pad_chunk = np.greater(iteri_this_location_block+windowsize, df_location.shape[0])
    is_last_chunk = np.greater_equal(iteri_this_location_block,end_iteri)
    # print(f"is_need_pad_chunk type {type(is_need_pad_chunk)}")
    # print(f"iteri_this_location_block is {iteri_this_location_block}")
    temparrs = []
    for key,bool_padned in enumerate(is_need_pad_chunk):
        if bool_padned:
            temparr = df_location.iloc[iteri_this_location_block[key]:,:].loc[:,train_cols]
        else:
            temparr = df_location.iloc[range(iteri_this_location_block[key], iteri_this_location_block[key]+windowsize),:]
        temparr = temparr.loc[:,train_cols]
        temparr = temparr.astype(final_numpy_data_dtype).to_numpy()
        if bool_padned:
            # print(f"temparr shape {temparr.shape} before vstack")
            padding_shape0 = windowsize-temparr.shape[0]
            empty_array_to_pad = np.full([padding_shape0,n_features], np.nan, dtype=final_numpy_data_dtype)
            temparr = np.vstack((temparr,empty_array_to_pad))
        # # print(f"temparr shape {temparr.shape} after vstack")
        temparr = np.expand_dims(temparr, axis=0)
        temparrs.append(temparr)
    temparr = np.concatenate(temparrs, axis=0)
    return temparr

def generate_training_h5(
        alldatadf:Union[dask.dataframe.DataFrame,pd.DataFrame],
        write_folder=pathlib.Path("R:\\"),
        n_sample_gap = 1,
        windowsize=windowsize,
    ):
    import h5py
    if isinstance(alldatadf,dask.dataframe.DataFrame):
        alldatadf = alldatadf.compute(**dask_compute_kwargs)
    timeseries_observations = []
    for locationcode in alldatadf['LocationCode'].unique():
        location_subset_df = alldatadf[alldatadf['LocationCode']==locationcode].reset_index(drop=True)
        nrows_location_subset_df = location_subset_df.shape[0]
        timeseries_observation = dask.array.arange(
            0,
            nrows_location_subset_df,
            n_sample_gap,
            chunks=nrows_location_subset_df
        ) #, chunks=dask_compute_kwargs['num_workers']
        timeseries_observation = timeseries_observation.map_blocks(
            rowindex_to_datadfarray,
            locationcode=locationcode,
            srcdf=location_subset_df,
            windowsize=windowsize,
            chunks=(1,windowsize,len(train_cols))
        )
        timeseries_observations.append(timeseries_observation)
        # print(timeseries_observation.compute(**dask_compute_kwargs).shape)
        print(f"locationcode {locationcode} done.")
    timeseries_observations_computed = dask.array.concatenate(timeseries_observations, axis=0).compute(**dask_compute_kwargs)
    print(f"final shape {timeseries_observations_computed.shape}")
    with h5py.File(write_folder/'traindata.h5', 'w') as f:
        f.create_dataset('timeseries_observations_computed', data=timeseries_observations_computed, dtype=final_numpy_data_dtype)


# %%
if __name__ == '__main__':
    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--directlyfromfile', type=ast.literal_eval, default=True, help="directlyfromfile")
    parser.add_argument('--write_folder', type=str, default="/content", help="write_folder")
    parser.add_argument('--n_sample_gap', type=int, default=1, help="n_sample_gap")
    parser.add_argument('--windowsize', type=int, default=30, help="windowsize")
    
    args = parser.parse_args()
    argp_dict = vars(args)
    print(f"argp_dict = {argp_dict}")
    
    # airqualitydata = get_air_pollution_data(join_locations=True,dftype='dd').compute(**dask_compute_kwargs)
    # competitiondata = custom_read_dfs('36_TrainingData', compute=True)
    # cwadf = custom_read_dfs('cwa')

    # 10min: empty percentage 0.78483, 595765 rows × 27 columns
    # 5min: empty percentage 0.78723, 1191513 rows × 27 columns
    # 2.5min: empty percentage 0.78919, 2383009 rows × 27 columns
    # 2min: empty percentage 0.789748, 1191513 rows × 27 columns
    # 1min: empty percentage 0.792039, 5957497 rows × 27 columns
    # argp_dict = {'directlyfromfile':False}
    alldatadf = complete_read_dfs(directlyfromfile=argp_dict['directlyfromfile'],set_index=False)
    alldatadf.to_parquet(current_dir/"alldatadf.parquet", engine="pyarrow")
    generate_training_h5(
        alldatadf=alldatadf,
        write_folder=pathlib.Path(argp_dict['write_folder']),
        n_sample_gap=argp_dict['n_sample_gap'],
        windowsize=argp_dict['windowsize'],
    )
    


    # import matplotlib.pyplot as plt
    # find duplicates to draw plot
    # duplicates_row = alldatadf_fulldatetime.loc[:,['LocationCode','DataTime']]#.reset_index(drop=True)
    # duplicates_row = duplicates_row.duplicated()
    # alldatadf_fulldatetime = alldatadf_fulldatetime[~duplicates_row].reset_index(drop=True)
    # duplicates_row_indices = duplicates_row.duplicated(subset=['LocationCode', 'basis_datetime'], keep=False)
    # duplicates_row_subset = duplicates_row[duplicates_row_indices]
    # .loc[duplicates_row,['LocationCode','DataTime']].reset_index(drop=True)
    # duplicates_row_subset_2 = duplicates_row_subset.copy()
    # duplicates_row_subset_2['DataTime'] += pd.Timedelta(-1, unit='min')
    # appendrows = pd.concat([
    #     (duplicates_row[duplicates_row].index+pd.Timedelta(-1, unit='min')).to_series().reset_index(drop=True),
    #     duplicates_row_subset['LocationCode'].reset_index(drop=True)
    # ], axis=1, ignore_index=True) \
    #     .rename(columns={0:'DataTime',1:'LocationCode'})
    # alldatadf.reset_index(drop=True).merge(appendrows, on=['DataTime','LocationCode'], how='inner')

    # trend plot
    # if True:
    #     testdf = alldatadf[alldatadf['LocationCode']==1]
    #     testdf = testdf['2024-01-01':'2024-01-05']
    #     testdf
    #     # Create a figure and a set of subplots
    #     fig, ax1 = plt.subplots()

    #     # Plot the first continuous variable
    #     ax1.plot(testdf['DateTime'], testdf['Sunlight(Lux)'], 'b-', label='Sunlight(Lux)')
    #     ax1.set_xlabel('Timestamp')
    #     ax1.set_ylabel('Sunlight(Lux)', color='b')
    #     ax1.tick_params('y', colors='b')

    #     # Create a second y-axis for the second continuous variable
    #     ax2 = ax1.twinx()
    #     ax2.plot(testdf['DateTime'], testdf['LuminationFactor'], 'r-', label='LuminationFactor')
    #     ax2.set_ylabel('LuminationFactor', color='r')
    #     ax2.tick_params('y', colors='r')

    #     # Add title and legend
    #     plt.title('Trend Plot with Dual Y-Axes')
    #     fig.tight_layout()  # Adjust layout to prevent overlap

    #     plt.show()
    
    # # missing value plot
    # if False:# Create a pivot table
    #     import plotly.express as px
    #     cwacols = ['StationPressure.Instantaneous','AirTemperature.Instantaneous','RelativeHumidity.Instantaneous','WindSpeed.Mean','Precipitation.Accumulation',]
    #     competitioncols = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)',
    #    'Humidity(%)', 'Power(mW)', 'Sunlight(Lux)']
    #     visualize_cols = ['Power(mW)']
    #     for pivot_target_column in visualize_cols:
    #         unpivot_table = alldatadf.loc[:,['LocationCode','DataTime',pivot_target_column]]
    #         unpivot_table = unpivot_table[~unpivot_table.duplicated()].reset_index(drop=True)
    #         unpivot_table[pivot_target_column] = unpivot_table[pivot_target_column].isna().astype('int')
    #         unpivot_table['LocationCode'] = unpivot_table['LocationCode'].astype('category')
            
    #         # pivot_table = pd.pivot_table(unpivot_table, index=['DataTime'], columns=['LocationCode'], values=[pivot_target_column], aggfunc="sum")
    #         # pivot_table.columns = pivot_table.columns.droplevel(0).tolist()
    #         # n_ypieces = 15
    #         # ypiece_size = len(pivot_table.index) // n_ypieces
    #         # y_first_elements = [pivot_table.index[i * ypiece_size].strftime('%Y-%m-%d') for i in range(n_ypieces)]

    #         fig = px.density_heatmap(
    #             unpivot_table,
    #             x='LocationCode',
    #             y='DataTime',
    #             z=pivot_target_column,
    #             histfunc='sum',
    #             title=pivot_target_column,
    #             nbinsy=100)
    #         # Configure number of ticks
    #         fig.update_xaxes(nticks=17)
    #         fig.update_yaxes(nticks=20)
    #         fig.show()

# find_neighbor_cwa_stations
# nice pairs
# 17	466990	
# 16	466990
# 15	466990
# 14	466990
# 13	466990 or C0Z180 -> 466990
# 12    466990
# 11    466990 or C0Z100 -> C0Z100
# 10    466990
# 9     466990
# 8     466990 or C0Z100 -> C0Z100
# 7     C0Z100
# 6     466990
# 5     466990 or C0Z100 -> C0Z100
# 4     C0T9G0(Pressure problem) or C0Z180(wind and pressure problem) -> ( 效能很差 ) -> 466990
# 3     466990 or C0Z100 -> C0Z100
# 2     C0Z100 or 466990 -> C0Z100
# 1     466990
# %%
# https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1
# https://github.com/ibm-granite/granite-tsfm/blob/ttm_v2_release/notebooks/tutorial/ttm_with_exog_tutorial.ipynb
# https://github.com/siang-chang/aidea-solar-energy-surplux