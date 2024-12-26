# %%
import dask.dataframe
import dask.bag
import requests
import urllib.parse
import pandas as pd
from typing import Literal
from IPython.display import display
import time
import random
import numpy as np
import pyarrow as pa
from pathlib import Path
from zoneinfo import ZoneInfo


def completedisplay(df:pd.DataFrame):
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
        display(df)

# 花蓮 466990 23.9751 121.6133
# 東華 C0Z100 23.8953121.5498
# 萬榮 C0Z200 23.7092 121.4201
# 鳳林山 C0T9G0 23.7353 121.4201
# 鳳林 C0Z160 23.7461 121.4534

# 裝置 1.風速(m/s) 2.大氣壓力(百帕) 3.溫度(攝氏) 4.相對濕度(百分比) 5.光照度(勒克斯) 6.發電量(瓦特)

# 東華
# 氣壓(hPa) 氣溫 相對溼度 風速(m/s) 最大瞬間風速(m/s) 降水量(mm)

currentfolder = Path(__file__).parent
cookie_string = 'CMP=u51q5uj9sbl2l7h57u6bifvpgg; CMPUT=1732371443; CMPEP=1732373249.020bc0868cd23cb; TS01665764=0107dddfef77c9ff25e613569b28d7c774066809484a782caf63bb163c7f184877f713d85768fdfc58195540e2ae5188e2100284f946419b15a9574dd0be52da3e9a743e8be7a02b995ced53d627e26555bd39b32eb4fabe4088be10323b3b52ad9cd4d89fc7ce95f2fc52b8de020d5da647b74de6e3460bcc5a64fe9524a4e3ed5a099ca80d1e21a3e0b12146d377911632f10293822da5acaf72a2eb96ff4ddce66b218a387a6869346e094e8cbac5bd9be8d180; TS5f4773d5027=08dc4bbcbbab2000751f4b2f794d1580cafbab881b374fa8e8e013f00f3a4fde1e6d28eb9804954508951e4a2f113000ddcf4089a6ccb534868257f63819fe3f943e1725e973394a73f009f819003b53d8c8f5cc66638e003984ccdbeb4cf006'
# %%
timezone = 'Asia/Taipei'
station_type = {
    'C0Z100':'auto_C0', #donghua
    # 'C0Z200':'auto_C0', #萬榮
    # 'C0Z160':'auto_C0', #鳳林
    # 'C0T9G0':'auto_C0', #鳳林山
    '466990':'cwb', #hualien
    # 'C0Z180':'auto_C0', #新城
    # 'C0T9E0':'auto_C0', #大坑
}
station_chname = {
    'C0Z100':'東華',
    # 'C0Z200':'萬榮',
    # 'C0Z160':'鳳林',
    # 'C0T9G0':'鳳林山',
    '466990':'花蓮',
    # 'C0Z180':'新城',
    # 'C0T9E0':'大坑',
}
    
weather_data_meta = [
    ("DataTime","datetime64[ns]"),
    ("StationPressure.Instantaneous","float64"),
    ("AirTemperature.Instantaneous","float64"),
    ("RelativeHumidity.Instantaneous","Int32"), # Int32
    ("WindSpeed.Mean","float64"),
    ("WindDirection.Mean","float64"),
    ("PeakGust.Maximum","float64"),
    ("PeakGust.Direction","Int32"),
    ("PeakGust.MaximumTime","datetime64[ns]"), #datetime
    ("Precipitation.Accumulation","float64"),
    ("Precipitation.MeltFlag","float64"),
    ("StationID","string[pyarrow]"),
    ("WindSpeed.TenMinutelyMaximum","float64"),
    ("WindDirection.TenMinutelyMaximum","float64"),
    ("GlobalSolarRadiation.Accumulation","float64"),
    ("Visibility.Instantaneous","float64"),
    ("TotalCloudAmount.Instantaneous","float64"),
]
string_to_pddtype_mapper = {
    "float64":pd.Float64Dtype(),
    "Int32":pd.Int32Dtype(),
    "datetime64[ns]":pd.DatetimeTZDtype(tz=ZoneInfo(timezone)),
    "string[pyarrow]":pd.StringDtype(storage="pyarrow"),
}
weather_data_meta_for_dd = list(
    (k,string_to_pddtype_mapper[v])
    for k,v in weather_data_meta)
weather_data_meta_for_pd = dict(weather_data_meta_for_dd)
pa_dtype_map = {
    "datetime64[ns]": pa.timestamp('ns', tz=timezone),
    "float64": pa.float64(),
    "Int64": pa.int64(),
    "Int32": pa.int32(),
    "string[pyarrow]": pa.string(),
    "object": pa.string()  # assuming object should be treated as string
}
weather_data_meta_for_pa = {
    name: pa_dtype_map[dtype]
    for name, dtype in weather_data_meta
}

# 空氣汙染下載資訊
# await fetch("https://airtw.moenv.gov.tw/cht/Query/InsValue.aspx", {
#     "credentials": "include",
#     "headers": {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0",
#         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8",
#         "Accept-Language": "zh-TW,en-US;q=0.7,en;q=0.3",
#         "Content-Type": "application/x-www-form-urlencoded",
#         "Sec-GPC": "1",
#         "Upgrade-Insecure-Requests": "1",
#         "Sec-Fetch-Dest": "document",
#         "Sec-Fetch-Mode": "navigate",
#         "Sec-Fetch-Site": "same-origin",
#         "Sec-Fetch-User": "?1",
#         "Priority": "u=0, i"
#     },
#     "referrer": "https://airtw.moenv.gov.tw/cht/Query/InsValue.aspx",

def compose_requestbody(*args):
    kwargs = args[0] # if len(args)>0 else {}
    # kwargs.setdefault("station", "donghua")
    # kwargs.setdefault("datetime_start", "2024-10-14T00:00:00")
    # kwargs.setdefault("datetime_end", "2024-10-15T23:59:59")
    # Literal['donghua','wangjong','hualien']
    # datetime_start:str='2024-10-14T00:00:00', datetime_end:str='2024-10-15T23:59:59'
    # station_code = {
    #     'donghua': {'stn_ID':'C0Z100','stn_type':'auto_C0'},
    #     'wangjong': {'stn_ID':'C0Z200','stn_type':'auto_C0'},
    #     'hualien': {'stn_ID':'466990','stn_type':'cwb'},
    # }
    global station_type
    req_template = [
        "https://codis.cwa.gov.tw/api/large_download_station?jtPageSize=999999999&jtStartIndex=0",
        {
            "headers": {
                "accept": "application/json, text/javascript, */*; q=0.01",
                "accept-language": "zh-TW,zh;q=0.9",
                "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
                "sec-ch-ua": "\"Google Chrome\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": "\"Windows\"",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "x-requested-with": "XMLHttpRequest",
                "origin": "https://codis.cwa.gov.tw",
                "referrer": "https://codis.cwa.gov.tw/LargeDownload",
            },
            "referrerPolicy": "strict-origin-when-cross-origin",
            "body": "type=hour&stn_ID=C0Z100&stn_type=auto_C0&start=2024-10-14T00%3A00%3A00&end=2024-10-15T23%3A59%3A59&item%5B%5D=StationPressure&item%5B%5D=AirTemperature&item%5B%5D=RelativeHumidity&item%5B%5D=WindSpeed%2CWindDirection&item%5B%5D=PeakGust&item%5B%5D=Precipitation",
            "method": "POST",
            "mode": "cors",
            "credentials": "include"
        }
    ]
    # 2024-10-14T00%3A00%3A00 urllib.parse.unquote(url_encoded_string)
    datetime_start_encoded = urllib.parse.quote(kwargs['datetime_start'])
    datetime_end_encoded = urllib.parse.quote(kwargs['datetime_end'])
    req_template[1]['body'] = req_template[1]['body'].replace('C0Z100', kwargs['station_ID'])
    req_template[1]['body'] = req_template[1]['body'].replace('auto_C0', station_type[kwargs['station_ID']])
    req_template[1]['body'] = req_template[1]['body'].replace('2024-10-14T00%3A00%3A00', datetime_start_encoded)
    req_template[1]['body'] = req_template[1]['body'].replace('2024-10-15T23%3A59%3A59', datetime_end_encoded)
    if kwargs['station_ID']=='466990':
        print(f"it's 466990")
        req_template[1]['body'] += "&item%5B%5D=GlobalSolarRadiation&item%5B%5D=Visibility&item%5B%5D=TotalCloudAmount"
    decoded_dict = urllib.parse.parse_qs(req_template[1]['body'])
    # cookie_string = 'CMP=dvqq46n8n2pp5pkd8bqu4vuf5k; CMPEP=1729178254.d992a86920a6ae2; CMPUT=1729176346; TS01665764=0107dddfef1479ba39de32246b1e93f93f2da4510f0a6eb35b0d3278944664ad087299dd8dc4650421b79138c07478340fbe7ed0ae8163ba0b14cf319b44fab877b0cf568b99c368e51670382fd0ffbdda6b492cd4f29d32b255d27bffcc7c86dc10a6a3135ee11622021e30ae20285637511189cda2a05096ff05d8fd7525f94629963570897d58b23e5304f7189efdff48d098b78f12f63cf28e0b24ac9d6393468c4a3a87698eb03b21ed9ccf9daed7f129ec1d; TS5f4773d5027=08dc4bbcbbab2000bdf246c9f17a7a096d077431f2b04310a680a8ef701769a59bdc9ec7055ffbaa08beccfa741130001d04d00f9f1220f564550b699bd5e1481f75348350ab06784a20e191de8676922f82873d10a34f017ceb293474183158'
    global cookie_string
    cookies = dict(x.split('=') for x in cookie_string.split('; '))
    time_to_sleep = random.uniform(1, 3)
    time.sleep(time_to_sleep)
    response = requests.post(req_template[0], data=decoded_dict, headers=req_template[1]['headers'], cookies=cookies)
    response_json = response.json()
    responsecookies = requests.utils.dict_from_cookiejar(response.cookies)
    cookies.update(responsecookies)
    cookie_string = "; ".join([f"{k}={v}" for k,v in cookies.items()])
    # print(f"response.cookies is {responsecookies}")
    if response.status_code!=200 or response_json['code'] in [404,500]:
        print(f"failed at {kwargs['station_ID']} {kwargs['datetime_start']} for {response_json}")
        return {}
    else:
        # print(f"{kwargs} finished.")
        return response_json

def process_cwa_json(data:dict)->pd.DataFrame:
    global weather_data_meta, timezone
    if len(data)==0:
        return pd.DataFrame()
    try:
        df = pd.json_normalize(data['data'],record_path=['dts'], meta=['StationID'], max_level=3)
    except Exception as e:
        print(f"data = {data}")
        raise(e)
    to_check_data = False
    for weather_data_meta_key,_ in weather_data_meta:
        if weather_data_meta_key not in df.columns:
            df[weather_data_meta_key] = pd.NA
            to_check_data = True
            # print(f"key {weather_data_meta_key} not in df.columns given StationID={df['StationID'].unique()} DataTime={df['DataTime'].unique()}")
    if to_check_data:
        completedisplay(df.drop(
            columns=['StationPressure.Instantaneous','AirTemperature.Instantaneous','RelativeHumidity.Instantaneous','WindSpeed.TenMinutelyMaximum','WindSpeed.Mean','WindDirection.TenMinutelyMaximum','WindDirection.Mean','PeakGust.Direction','PeakGust.Maximum','Precipitation.Accumulation','PeakGust.MaximumTime','Precipitation.MeltFlag'],
            errors='ignore'
            ).head(n=10))
    df['DataTime'] = pd.to_datetime(df['DataTime'])
    df['DataTime'] = df['DataTime'].dt.tz_localize(timezone)
    # request_datetime_start = pd.Series([actualrequests['datetime_start']])
    # request_datetime_start = pd.to_datetime(request_datetime_start)
    # request_datetime_start = request_datetime_start.dt.tz_localize('Asia/Taipei')
    # request_datetime_start-= pd.Timedelta(8, unit='h')
    # display(request_datetime_start)
    df["PeakGust.MaximumTime"] = df["PeakGust.MaximumTime"].astype(pd.StringDtype(storage="pyarrow"))
    df["PeakGust.MaximumTime"][~df["PeakGust.MaximumTime"].str.contains("T")] = pd.NA #np.nan
    df["PeakGust.MaximumTime"] = pd.to_datetime(df['PeakGust.MaximumTime'])
    df["PeakGust.MaximumTime"] = df["PeakGust.MaximumTime"].dt.tz_localize(timezone)
    # df['DataTime_hourofday'] = df['DataTime'].dt.hour
    # df['DataTime_dayofyear'] = df['DataTime'].dt.day_of_year
    for colname in df.columns:
        df[colname] = df[colname].astype( weather_data_meta_for_pd[colname] )
    return df

if False:
    testcase = {"station_ID": "C0Z100",
                "datetime_start": "2024-03-29T00:00:00",
                "datetime_end": "2024-03-30T23:59:59"}
    test_fetched_dict = compose_requestbody(testcase)
    testdf = process_cwa_json(test_fetched_dict, testcase)
    display(
        testdf
    )
# %%
if __name__ == '__main__':
    end = '2024-11-20'
    date_start_series = pd.date_range(start='2024-01-01', end=end, freq='30D')
    date_start_series_str = date_start_series.strftime('%Y-%m-%dT%H:%M:%S')
    date_end_series = date_start_series + pd.Timedelta(30*24*60*60-1, unit='s')
    date_end_series_str = date_end_series.strftime('%Y-%m-%dT%H:%M:%S')
    all_parameter_df = pd.merge(
        left=pd.DataFrame(list(station_type.keys()), columns=['station_ID']),
        right=pd.concat([pd.Series(x) for x in [date_start_series_str, date_end_series_str]], axis=1).rename(columns={0: "datetime_start", 1: "datetime_end"}),
        how='cross'
    )
    # all_input_stations_datetimes = dask.bag.from_sequence(all_parameter_df.to_dict(orient='records'))
    # all_input_stations_datetimes = dask.bag.map(compose_requestbody, all_input_stations_datetimes)
    # all_input_stations_datetimes = all_input_stations_datetimes.compute(num_workers=1)
    all_input_stations_datetimes = [compose_requestbody(d) for d in all_parameter_df.to_dict(orient='records')]
    # %%
    all_input_stations_datetimes_dd = dask.bag.from_sequence(all_input_stations_datetimes)
    df = dask.dataframe.from_map(
        process_cwa_json,
        all_input_stations_datetimes_dd,
        meta=weather_data_meta_for_dd,
    )

    # %%
    df.to_parquet(
        currentfolder/'cwa',
        engine='pyarrow',
        write_index=False,
        overwrite=True,
        schema=weather_data_meta_for_pa,
        name_function=lambda n: f'part.{n:03}.parquet',
    )
    # df.to_parquet(currentfolder/'cwadata')
    # %%
