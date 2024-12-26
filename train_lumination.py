# %%
import pandas as pd
from pathlib import Path
from typing import Literal,Union
import math
import numpy as np
import torch
# from torchmetrics.functional.regression import mean_absolute_percentage_error
from IPython.display import display
import dask.array
from sklearn.model_selection import KFold
import importlib
import get_data
import find_elevation_angle_from_astral
importlib.reload(find_elevation_angle_from_astral)
locationcode_dtype = 'int8'
min_time_interval = '10min'
timezone = "Asia/Taipei"

locationCode_coordinate_pairs = {
    1:[23.8994444444444,121.544444444444],
    2:[23.8997222222222,121.544722222222],
    3:[23.8997222222222,121.545],
    4:[23.8994444444444,121.544444444444],
    5:[23.8994444444444,121.544722222222],
    6:[23.8994444444444,121.544444444444],
    7:[23.8994444444444,121.544444444444],
    8:[23.8997222222222,121.545],
    9:[23.8994444444444,121.544444444444],
    10:[23.8994444444444,121.544444444444],
    11:[23.8997222222222,121.544722222222],
    12:[23.8997222222222,121.544722222222],
    13:[23.8977777777778,121.539444444444],
    14:[23.8977777777778,121.539444444444],
    15:[24.0091666666667,121.617222222222],
    16:[24.0088888888889,121.617222222222],
    17:[24.0105807,121.6133654],
}
locationCode_surface_azimuth_pairs = {
    1:181,
    2:175,
    3:180,
    4:161,
    5:208,
    6:208,
    7:172,
    8:219,
    9:151,
    10:223,
    11:131,
    12:298,
    13:249,
    14:197,
    15:127,
    16:82,
    17:180,
}
floor_altitudes = {
    1:18.0,
    2:18.0,
    3:18.0,
    4:18.0,
    5:18.0,
    6:18.0,
    7:18.0,
    8:10.8,
    9:10.8,
    10:3.6,
    11:3.6,
    12:3.6,
    13:18.0,
    14:18.0,
    15:3.6,
    16:3.6,
    17:7.2,
}
location_surface_azimuth_pairs_df = pd.DataFrame.from_dict(locationCode_surface_azimuth_pairs, orient='index').reset_index().rename(columns={'index':'LocationCode',0:'surface_azimuth'})
location_altitude_pairs_df = pd.DataFrame.from_dict(floor_altitudes, orient='index').reset_index().rename(columns={'index':'LocationCode',0:'altitude'})
location_coordinate_pairs_df = pd.DataFrame.from_dict(locationCode_coordinate_pairs, orient='index').reset_index(drop=False).rename(columns={'index':'LocationCode',0:'latitude',1:'longitude'})
location_coordinate_pairs_df['LocationCode'] = location_coordinate_pairs_df['LocationCode'].astype(locationcode_dtype)
location_coord_azimuth_pairs_df = location_surface_azimuth_pairs_df.merge(location_coordinate_pairs_df, how='inner').merge(location_altitude_pairs_df)
location_coord_azimuth_pairs_df = location_coord_azimuth_pairs_df.assign(surface_tilt=20)
# %%

# %%

standard_coordinate = {'latitude':23.8927795,'longitude':121.5415863}
# t = pvlib.iotools.get_pvgis_tmy(standard_coordinate['latitude'], standard_coordinate['longitude'])
# t[0]

# mc = ModelChain(system, location)
# 'ghi'
# global horizontal irradiance
# 'dni'
# direct normal irradiance
# 'dhi'
# diffuse horizontal irradiance


def get_lumination_factor(input_datetime, latitude, longitude):
    observer = find_elevation_angle_from_astral.Observer()
    observer.latitude = latitude
    observer.longitude = longitude
    factor = math.sin(math.radians(
        find_elevation_angle_from_astral.elevation(
            observer=observer,
            dateandtime=input_datetime,
            with_refraction=True
        )
    ))
    return factor if factor>=0 else 0
get_lumination_factor_vectorized = np.vectorize(get_lumination_factor)


# %%
# https://www.aicup.tw/ai-cup-2024-competition
# 光照度低於20lux會進入休眠 17:00~17:30
# 花蓮縣壽豐鄉校本部23.8927795,121.5415863
# 東華 C0Z100
# 花蓮市區美崙校區24.0102456,121.6175005
# 花蓮 466990 新城 C0Z180、
# 花蓮縣鳳林鎮 (兆豐農場旁邊23.8019603)
# 萬榮 C0Z200 鳳林山 C0T9G0 鳳林 C0Z160
# https://www.perplexity.ai/search/dang-di-tai-yang-shi-jian-loca-Y7g8l_oAQjaBaLw5LTgNkw
# https://github.com/wenjiedu/awesome_imputation
# https://github.com/qingsongedu/time-series-transformers-review?tab=readme-ov-file

# https://github.com/aaghamohammadi/pysolorie
# 太陽能發電板的經緯度座標、面朝方向與發電量之間的關係可以通過數學公式來表達。以下是關鍵概念及其公式的詳細說明。

# ## 1. 基本概念

# - **經緯度**：影響太陽能板接收陽光的角度，尤其是緯度會直接影響最佳傾斜角度。
# - **面朝方向**（方位角）：指太陽能板相對於正南方向的角度，通常以度數表示。朝南（0°）通常是最佳方向。
# - **傾斜角**：太陽能板相對於水平面的傾斜角度，會影響日照量和發電效率。

# ## 2. 數學表達式

# ### 2.1 傾斜角計算

# 傾斜角 ($$ \theta $$) 可以根據以下公式計算：

# $$
# \theta = \text{緯度} + \text{太陽赤緯}
# $$

# 其中，太陽赤緯 ($$ \delta $$) 是太陽相對於地球赤道的角度，隨季節變化而變化。

# ### 2.2 方位角計算

# 方位角 ($$ A $$) 的計算公式為：

# $$
# A = (\text{負荷峰值時刻} - 12) \times 15 + (\text{經度} - 116)
# $$

# 這個公式用於調整太陽能板的方位，使其在一天中獲得最大發電量。

# ### 2.3 發電量估算

# 發電量 ($$ P $$) 可以用以下公式估算：

# $$
# P = E \cdot A_{effective} \cdot H
# $$

# 其中：
# - $$ E $$ 是每平方米的光伏效率（通常取決於面板類型）。
# - $$ A_{effective} $$ 是有效接收面積，考慮到傾斜角和方位角的影響。
# - $$ H $$ 是每日有效日照小時數，取決於地理位置和季節。

# ### 2.4 有效接收面積計算

# 有效接收面積 ($$ A_{effective} $$) 可以用以下公式計算：

# $$
# A_{effective} = A_{total} \cdot \cos(\theta) \cdot \cos(A)
# $$

# 其中：
# - $$ A_{total} $$ 是太陽能板的總面積。

# ## 3. 綜合關係

# 將以上公式綜合，可以得出一個關於經緯度、方位角、傾斜角與發電量之間的關係：

# $$
# P = E \cdot A_{total} \cdot H \cdot \cos(\theta) \cdot \cos(A)
# $$

# 這個公式顯示了如何根據不同的地理位置（經緯度）、安裝方向（方位角）和傾斜程度（傾斜角）來計算太陽能發電板的預期發電量。透過這些數學式，可以更好地設計和優化太陽能系統，以提高其發電效率。

# Citations:
# [1] https://zh-tw.shieldenchannel.com/blogs/solar-panels/ngle-and-orientation-for-solar-panels
# [2] https://solarchanghua.com/solar-info/popular-services/knowledge/122-knowledge.html
# [3] https://www.freeroof.com.tw/uncategorized/position/
# [4] https://learnenergy.tw/index.php?caid=7&id=484&inter=knowledge
# [5] https://rdnet.taichung.gov.tw/media/725358/%E9%81%8B%E7%94%A8%E6%97%A5%E7%85%A7%E9%81%AE%E8%94%BD%E5%8F%8A%E8%BC%BB%E5%B0%84%E9%87%8F%E8%A9%95%E4%BC%B0%E5%B1%8B%E9%A0%82%E5%9E%8B%E5%A4%AA%E9%99%BD%E8%83%BD%E7%99%BC%E5%B1%95%E6%BD%9B%E5%8A%9B-%E4%BB%A5%E8%87%BA%E4%B8%AD%E5%B8%82%E7%82%BA%E4%BE%8B.pdf
# [6] https://www.hengs.com/QA-solar-power-plants.html
# [7] https://cn.linkedin.com/pulse/%E5%AE%89%E8%A3%85%E5%A4%AA%E9%98%B3%E8%83%BD%E6%9D%BF%E7%9A%84%E6%9C%80%E4%BD%B3%E6%9C%9D%E5%90%91%E4%B8%8E%E8%A7%92%E5%BA%A6%E5%A4%AA%E9%98%B3%E8%83%BD%E6%9D%BF%E5%AE%89%E8%A3%85%E8%A7%92%E5%BA%A6%E5%A4%AA%E9%98%B3%E8%83%BD%E7%94%B5%E6%B1%A0%E6%9D%BF%E6%9C%9D%E5%90%91%E5%A4%AA%E9%98%B3%E8%83%BD%E5%85%89%E4%BC%8F%E6%9D%BF%E5%8F%91%E7%94%B5%E6%95%88%E7%8E%87-richard-li-2av7e

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
class findlatitudeModel(torch.nn.Module):
    def __init__(self):
        super(findlatitudeModel, self).__init__()
        self.luminous_max_basis = torch.nn.Parameter(
            torch.tensor(1.0)
        )
        self.location_adjustments_x_layer_0 = torch.nn.LazyLinear(1).double()
        self.location_adjustments_y_layer_0 = torch.nn.LazyLinear(1).double()
        self.relu_m = torch.nn.ReLU()
        self.torchpi = torch.tensor(math.pi, requires_grad=False).to(device)
        self.localstandard_meridian = torch.tensor(15.0*8, requires_grad=False).to(device)
        self.is_parameters_initialized = False
        self.n_location_feature_levels = 16
        self.linear_for_latitude_adjustment = torch.nn.LazyLinear(1).double()
        self.linear_for_longitude_adjustment = torch.nn.LazyLinear(1).double()
        self.init_latitude = torch.tensor(23.8927795, requires_grad=False).to(device)
        self.init_longitude = torch.tensor(121.5415863, requires_grad=False).to(device)

    def init_parameters(self, in_dim):
        # self.latitude = torch.nn.parameter.Parameter(
        #     torch.full(size=(in_dim,1), fill_value=23.8927795, requires_grad=True, device=device)
        # )
        # self.longitude = torch.nn.parameter.Parameter(
        #     torch.full(size=(in_dim,1), fill_value=121.5415863, requires_grad=True, device=device)
        # )
        pass

    def get_lumination_with_astronomy_from_array(self, srcdf):
        returnnumpy = self.get_lumination_with_astronomy(srcdf[:,0], srcdf[:,1], srcdf[:,2], srcdf[:,3], conver_to_tensor_first=True)
        # print(f"1 returnnumpy type {type(returnnumpy)} shape {returnnumpy.shape}")
        returnnumpy = returnnumpy.cpu().detach().numpy()
        # print(f"2 returnnumpy type {type(returnnumpy)} shape {returnnumpy.shape}")
        return returnnumpy

    # def get_lumination_with_astronomy(self, dayofyr, hour_of_day, latitude, longitude, conver_to_tensor_first=False)->torch.tensor:
    #     if conver_to_tensor_first:
    #         # print(f"types: {type(dayofyr)}, {type(hour_of_day)}, {type(latitude)}, {type(longitude)},")
    #         dayofyr = torch.tensor(dayofyr, requires_grad=False, device=device, dtype=torch.float64)
    #         hour_of_day = torch.tensor(hour_of_day, requires_grad=False, device=device, dtype=torch.float64)
    #         latitude = torch.tensor(latitude, requires_grad=False, device=device, dtype=torch.float64)
    #         longitude = torch.tensor(longitude, requires_grad=False, device=device, dtype=torch.float64)
    #     with torch.no_grad():
    #         # 太陽的赤緯角
    #         declination = torch.add(dayofyr,10.0)
    #         declination = 360.0/365.0*(declination) #n*1
    #         declination = torch.deg2rad(declination)
    #         declination = -23.44*torch.cos(declination)
    #         declination_sin = torch.sin(torch.deg2rad(declination)) #n*1
    #         declination_cos = torch.cos(torch.deg2rad(declination)) #n*1
    #         # 均時差（EoT）
    #         eot_b = ((dayofyr-81.0)*360.0/365.0) #n*1
    #         eot_b_radian = torch.deg2rad(eot_b)
    #         eot = 9.87 * torch.sin( torch.deg2rad(2.0*eot_b) )
    #         eot = eot-7.53*torch.cos( eot_b_radian )
    #         eot = eot-1.5*torch.sin( eot_b_radian )*((dayofyr-81.0)/365.0)
    #     time_corrector = 4*(longitude-self.localstandard_meridian)+eot #n*1

    #     # 當地太陽時 local solar time
    #     local_solar_time = hour_of_day+(time_corrector/60.0)
    #     # 時角
    #     hour_angle = 15.0*(local_solar_time-12.0)
    #     hour_angle_cos = torch.cos(torch.deg2rad(hour_angle))

    #     # 太陽高度角
    #     solar_elevation_angle = torch.arcsin(
    #         hour_angle_cos*declination_cos*torch.cos(torch.deg2rad(latitude)) +
    #         declination_sin*torch.sin(torch.deg2rad(latitude))
    #     )
    #     solar_elevation_angle_sin = torch.sin(solar_elevation_angle) #這是一個反應環境光照強度的係數
    #     solar_elevation_angle_sin = torch.clamp(solar_elevation_angle_sin, min=0.0)

    #     return solar_elevation_angle_sin

    def get_lumination_with_astronomy(self, dayofyr, hour_of_day, latitude, longitude, conver_to_tensor_first=False)->torch.tensor:
        if conver_to_tensor_first:
            # print(f"types: {type(dayofyr)}, {type(hour_of_day)}, {type(latitude)}, {type(longitude)},")
            dayofyr = torch.tensor(dayofyr, requires_grad=False, device=device, dtype=torch.float64)
            hour_of_day = torch.tensor(hour_of_day, requires_grad=False, device=device, dtype=torch.float64)
            latitude = torch.tensor(latitude, requires_grad=False, device=device, dtype=torch.float64)
            longitude = torch.tensor(longitude, requires_grad=False, device=device, dtype=torch.float64)
        with torch.no_grad():
            # 太陽的赤緯角
            declination = torch.subtract(dayofyr,81.0)
            declination = torch.deg2rad(torch.tensor(360.0))/365.0*(declination) #n*1
            declination = 23.45*torch.sin(declination) #in degree
            declination_sin = torch.sin(torch.deg2rad(declination)) #n*1
            declination_cos = torch.cos(torch.deg2rad(declination)) #n*1
            # 均時差（EoT）
            eot_b = torch.subtract(dayofyr,81.0)
            eot_b = torch.mul(eot_b,360.0)
            eot_b = torch.div(eot_b,365.0)
            eot_b = torch.deg2rad(eot_b)
            eot = 9.87 * torch.sin( eot_b*2 )
            eot = eot-7.53*torch.cos( eot_b )
            eot = eot-1.5*torch.sin( eot_b )
        time_corrector = 4.0*(longitude-self.localstandard_meridian)+eot #n*1

        # 當地太陽時 local solar time
        local_solar_time = hour_of_day+(time_corrector/60.0)
        # 時角
        hour_angle = 15.0*(local_solar_time-12.0) # in degree
        hour_angle_cos = torch.cos(torch.deg2rad(hour_angle))

        # 太陽高度角
        solar_elevation_angle = torch.arcsin(
            hour_angle_cos*declination_cos*torch.cos(torch.deg2rad(latitude)) +
            declination_sin*torch.sin(torch.deg2rad(latitude))
        ) # in radian
        # 太陽天頂角
        solar_zenith_angle = torch.deg2rad(torch.tensor(90.0))-solar_elevation_angle # in radian
        solar_zenith_angle_cos = torch.cos(solar_zenith_angle) #這是一個反應環境光照強度的係數
        lumination_factor = torch.clamp(solar_zenith_angle_cos, min=0.0)

        return lumination_factor

    def forward(self, x):
        if not self.is_parameters_initialized:
            self.init_parameters(in_dim=self.n_location_feature_levels)
            self.is_parameters_initialized = True
        nrows = x.shape[0]
        dayofyr = x[:,0]
        hour_of_day = x[:,1]
        location_features_begin_end = range(2,2+self.n_location_feature_levels,1)
        location_features = x[:,list(location_features_begin_end)]

        latitude = self.init_latitude*location_features #緯度未知 n*16
        longitude = self.init_longitude*location_features #經度未知 n*16
        latitude = self.linear_for_latitude_adjustment(latitude) #n*1
        latitude = self.relu_m(latitude) #n*1
        latitude = latitude[:,0] #n
        longitude = self.linear_for_longitude_adjustment(longitude) #n*1
        longitude = self.relu_m(longitude) #n*1
        longitude = longitude[:,0] #n

        lumination_factor_from_astronomy = self.get_lumination_with_astronomy(dayofyr, hour_of_day, latitude, longitude) #n*1
        lumination_factor_from_astronomy = lumination_factor_from_astronomy.view(-1,1)

        # location adjustments on lumination
        lumination = lumination_factor_from_astronomy*( self.relu_m(self.location_adjustments_y_layer_0(location_features)) ) 
        lumination = lumination+self.relu_m(self.location_adjustments_x_layer_0(location_features))
        outputs = lumination

        return outputs
# %%
model = findlatitudeModel().to(device)
# %%
# test_dayofyr = torch.full(size=(24,1),fill_value=1, dtype=torch.float64, device=device)
# test_hour_of_day = torch.tensor([i for i in range(0,24,1)], dtype=torch.float64, device=device).view(24,1)
# test_latitude = torch.full(size=(24,1),fill_value=23.5, dtype=torch.float64, device=device)
# test_longitude = torch.full(size=(24,1),fill_value=120.0, dtype=torch.float64, device=device)
# model.get_lumination_with_astronomy(test_dayofyr, test_hour_of_day, test_latitude, test_longitude) #, luminous_max_basis=torch.tensor(1.0)
# %%
if __name__ == '__main__':
    pass
    # criterion = torch.nn.MSELoss(reduction='mean')
    # optimizer = torch.optim.AdamW(model.parameters()) #torch.optim.SGD(model.parameters(), lr=1e-8)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
    # srcdf = get_data.get_data_of_a_folder(type="36_TrainingData")
    # x = srcdf.filter(regex=("LocationCode_.+"), axis=1)
    # x = pd.concat([srcdf.loc[:,['dayofyear','hourofday']], x], axis=1)
    # k_of_KFold = 5
    # kf = KFold(n_splits=k_of_KFold, shuffle=True, random_state=1)
    # kf.get_n_splits(x.values)
    # x_tensor = torch.tensor(x.values, dtype=torch.float64)
    # y_tensor = torch.tensor(srcdf['Sunlight(Lux)'].values, dtype=torch.float64).view(-1,1)
    # # mean_abs_percentage_error = MeanAbsolutePercentageError()
    # # %%
    # # print(f"y dtype {y.dtype}")
    # for epoch in range(20000):
    #     losses = torch.empty((0, 10), dtype=torch.float64)
    #     all_predictions = []
    #     all_targets = []
    #     testset_metric = []
    #     for fold_i, (train_index, test_index) in enumerate(kf.split(x.values)):
    #         # print(f"now in fold_i {fold_i}")
    #         # Forward pass: Compute predicted y by passing x to the model
    #         inputs = x_tensor[train_index,:].to(device)
    #         y_pred = model(inputs).to(device)

    #         # Compute and print loss
    #         loss = criterion(y_pred, y_tensor[train_index,:].to(device))
    #         # print(f"y_pred[0] {y_pred[0,0]} loss_cpu {loss_cpu}")
    #         if epoch % 10 == 0:
    #             with torch.no_grad():
    #                 testoutputs = model(x_tensor[test_index,:].to(device))
    #                 all_predictions.append(testoutputs)
    #                 all_targets.append(y_tensor[test_index,:].to(device))
    #             if fold_i == k_of_KFold-1:
    #                 all_predictions = torch.cat(all_predictions, dim=0)
    #                 all_targets = torch.cat(all_targets, dim=0)
    #                 mspe_metric = mean_absolute_percentage_error(all_predictions, all_targets)
    #         # Zero gradients, perform a backward pass, and update the weights.
    #         optimizer.zero_grad()
    #         if not torch.isnan(loss):
    #             losses = torch.cat((losses, loss), 0)
    #             loss.backward()
    #             optimizer.step()
    #         else:
    #             break
    #     if torch.any(torch.isnan(torch.tensor(losses))) or torch.isnan(loss):
    #         break
    #     loss_avg = torch.mean(losses) #sum(losses) / len(losses)
    #     scheduler.step(loss_avg)
    #     print(f"epoch {epoch} Fold {fold_i} loss_avg {loss_avg} loss.items {loss.item()}")
    #     if epoch % 10 == 0:
    #         print(f"test eval metrics: {mspe_metric}")

    # # %%
    # for name, param in model.named_parameters():
    #     print( name, param.data)
    # %%
