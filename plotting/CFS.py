import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata, Rbf
from scipy.stats import chi2_contingency,pearsonr
from scipy.interpolate import griddata, RBFInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy import linalg
import pandas as pd
import seaborn as sns
import pyproj
from pyproj import Geod
import os
from scipy.interpolate import RegularGridInterpolator

# 区域参数
LAT_MIN, LAT_MAX = 26, 28
LON_MIN, LON_MAX = 102, 104
GRID_RESOLUTION = 0.005  # 网格分辨率(度)

# 物理参数
G = 30e9  # 剪切模量 (Pa)
NU = 0.25  # 泊松比
MU_PRIME =0.4 # 约 0.577
  # 有效摩擦系数
LAMBDA = 2 * G * NU / (1 - 2 * NU)  # 拉梅常数
RHO_WATER = 1000  # 水密度 (kg/m³)
GRAVITY = 9.8  # 重力加速度 (m/s²)
# 分层黏滞系数 (深度, 黏滞系数) - 单位: m, Pa·s
VISCOSITY_LAYERS = [
    (0, 1e21),    # 上地壳 (0-15km)
    (15e3, 1e19), # 中地壳 (15-30km)
    (30e3, 1e19)# 下地壳 (>30km)
]
# 初始化坐标投影转换器 (UTM Zone 48N for 102-103.3°E)
TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

def project_coords(lons, lats):
    """将经纬度投影为UTM直角坐标（单位：米）"""
    x, y = TRANSFORMER.transform(lons, lats)
    return np.array(x), np.array(y) 


# 1. 数据准备 - 这里需要替换为实际数据
def load_data(
    my_catalog,
    gnss_file,
    dilatation_file,
    shear_strain_file,
    rotation_file,
    groundwater_file  # 新增地下水位数据参数
):
    """
    加载所有必要数据（示例结构）
    返回:
        earthquakes: (N, 4) array [lon, lat, depth, magnitude]
        gnss_data: (M, 5) array [lon, lat, ve, vn, sig]
        dilatation: (K, 3) array [lon, lat, value]
        shear_strain: (K, 3) array [lon, lat, value]
        rotation: (K, 3) array [lon, lat, value]
        groundwater: (P, 3) array [lon, lat, value]  # 新增地下水位数据
    """
    # 实际应用中应替换为真实数据加载代码

    
    # 读取hypoDD_v2_2208.loc文件（假设是空格分隔的固定宽度格式）
    column_names = ['lat', 'lon', 'depth', 'year', 'month', 'day', 'hour', 'minute', 'second', 'magnitude','time']
    my_catalog = pd.read_csv(my_catalog, sep='\t', header=0, names=column_names)
    earthquakes = my_catalog[['lon', 'lat', 'depth', 'magnitude']].values
    
    # 读取GNSS数据
    gnss_df = pd.read_excel(gnss_file)
    gnss_data = gnss_df[['Longitude', 'Latitude', 'Ve', 'Vn', 'dVe','dVn']].values
    # 过滤异常值
    speed = np.sqrt(gnss_data[:,2]**2 + gnss_data[:,3]**2)
    valid_mask = (speed < 100)  # 速度<100 mm/yr
    gnss_data = gnss_data[valid_mask]
    mask = (
    (gnss_data[:,0] >= LON_MIN) & (gnss_data[:,0] <= LON_MAX) &
    (gnss_data[:,1] >= LAT_MIN) & (gnss_data[:,1] <= LAT_MAX)
    )

    gnss_data = gnss_data[mask]
    
    print(f"加载GNSS数据: {len(gnss_data)} 个站点 (过滤后)")
    print(f"GNSS Ve 范围: {np.min(gnss_data[:,2]):.2f} - {np.max(gnss_data[:,2]):.2f} mm/yr")
    print(f"GNSS Vn 范围: {np.min(gnss_data[:,3]):.2f} - {np.max(gnss_data[:,3]):.2f} mm/yr")

        
    # 3. 网格数据加载函数
    def load_grid(filepath, col_name, scale=1e-9):
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            if 'Longitude' in df.columns and 'Latitude' in df.columns and col_name in df.columns:
                mask = (
                (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX) &
                (df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX)
                )
                df = df[mask]
                values = df[col_name].values * scale  # 单位转换
                
                print(f"加载 {col_name} 数据: {len(df)} 个点 (过滤后)")
                return np.column_stack([df[['Longitude', 'Latitude']].values, values])
            else:
                raise KeyError(f"{filepath} 缺少Longitude/Latitude/{col_name}列")
        else:
            raise FileNotFoundError(f"文件不存在: {filepath}")
    
    try:
        dilatation = load_grid(dilatation_file, 'Dilation rate (10e-7)')
        print(f"加载膨胀率数据: {len(dilatation)} 个点")
    except Exception as e:
        print(f"膨胀率加载失败: {str(e)}")
        dilatation = np.zeros((1, 3))  # 空数据
    
    try:
        shear_strain = load_grid(shear_strain_file, 'maxx(10e-7)')
        print(f"加载剪应变率数据: {len(shear_strain)} 个点")
    except Exception as e:
        print(f"剪应变率加载失败: {str(e)}")
        shear_strain = np.zeros((1, 3))
    
    try:
        rotation = load_grid(rotation_file, 'rotation rate (10e-7)')
        print(f"加载旋转率数据: {len(rotation)} 个点")
    except Exception as e:
        print(f"旋转率加载失败: {str(e)}")
        rotation = np.zeros((1, 3))
    # 读取地下水位数据
    column_names = ['Date','Waterlevel']
    groundwater = pd.read_csv(groundwater_file, delim_whitespace=True, header=None,names= column_names)
    
    groundwater['Date'] = pd.to_datetime(groundwater['Date'], format='%Y%m%d')
    print(f"地下水位数据时间范围: {groundwater['Date'].min()} 至 {groundwater['Date'].max()}")
    groundwater = groundwater[['Date','Waterlevel']].values
    print(f"加载地下水位数据: {len(groundwater)} 个时间点")
    return earthquakes, gnss_data, dilatation, shear_strain, rotation,groundwater

# 2. 数据预处理
def preprocess_data(gnss_data, dilatation, shear_strain, rotation,groundwater):
    """处理并网格化所有数据"""
    # 创建统一网格
    grid_lons = np.arange(LON_MIN, LON_MAX + GRID_RESOLUTION, GRID_RESOLUTION)
    grid_lats = np.arange(LAT_MIN, LAT_MAX + GRID_RESOLUTION, GRID_RESOLUTION)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
  
    # 投影网格坐标
    grid_x, grid_y = project_coords(grid_lon.flatten(), grid_lat.flatten())
    grid_x = grid_x.reshape(grid_lon.shape)
    grid_y = grid_y.reshape(grid_lat.shape)
    
    print(f"UTM坐标范围: X={np.min(grid_x)}-{np.max(grid_x)} m, Y={np.min(grid_y)}-{np.max(grid_y)} m")
    # 正常应为几百公里量级（如x=500~600km，y=2800~2900km）
    
    # 修改这里：计算网格间距（米）作为一维数组
    dx = np.mean(np.diff(grid_x[0, :]))  # 经度方向的平均间距
    dy = np.mean(np.diff(grid_y[:, 0]))  # 纬度方向的平均间距
    # 投影GNSS站点坐标
    gnss_x, gnss_y = project_coords(gnss_data[:,0], gnss_data[:,1])
    
    # 使用RBF插值GNSS速度场（考虑误差权重）
    print("插值GNSS速度场...")
    
    from scipy.interpolate import Rbf

    rbf_ve = Rbf(gnss_x, gnss_y, gnss_data[:,2], function='multiquadric', smooth=0.1)
    rbf_vn = Rbf(gnss_x, gnss_y, gnss_data[:,3], function='multiquadric', smooth=0.1)

    ve_grid = rbf_ve(grid_x, grid_y)
    vn_grid = rbf_vn(grid_x, grid_y)


    valid_mask = ~np.isnan(ve_grid)
    print(f"有效插值点比例: {np.sum(valid_mask)/ve_grid.size:.1%}")
    # 建议添加一个可选的RBF fallback策略
    if np.sum(valid_mask) < 0.5 * ve_grid.size:
        from scipy.interpolate import Rbf
        rbf_ve = Rbf(gnss_x, gnss_y, gnss_data[:,2], function='linear')
        ve_grid = rbf_ve(grid_x, grid_y)
            
        
    # 插值其他网格数据
    def interp_grid(data):
        if len(data) > 1:
            x, y = project_coords(data[:,0], data[:,1])
            rbf = Rbf(x, y, data[:,2], function='multiquadric', smooth=0.1)
            return rbf(grid_x, grid_y)
        else:
            return np.zeros_like(grid_x)
    
    dilatation_grid = interp_grid(dilatation)
    shear_strain_grid = interp_grid(shear_strain)
    rotation_grid = interp_grid(rotation)
    
    # 验证旋转率范围
    print(f"旋转率范围: {rotation_grid.min():.2e}-{rotation_grid.max():.2e} rad/yr")
    
    # 处理水位数据    
    # 计算水位变化率
    dates = pd.to_datetime(groundwater[:,0].astype(str)).values.astype('datetime64[ns]')
    levels = groundwater[:,1].astype(float)
    # 计算总时间跨度（年）
    days_diff = (dates[-1] - dates[0]).astype('timedelta64[D]').astype(float)
    time_span_years = days_diff / 365.25
    # 计算年变化率
    level_diff = levels[-1] - levels[0]
    yearly_rate = level_diff / time_span_years
    
    
    # 空间衰减模型
    well_x, well_y = project_coords(103.02, 26.75)
    dist = np.sqrt((grid_x-well_x)**2 + (grid_y-well_y)**2)
    gw_grid = yearly_rate * np.exp(-dist/5000)  # 5km衰减半径
    
    print(f"井位的水位变化: {yearly_rate:.2f} m/yr")
    print(f"level_diff: {level_diff:.2f} m, time_span_years: {time_span_years:.2f} years, yearly_rate: {yearly_rate:.2f} m/yr")
    print(f"水位范围: {np.min(groundwater[:,1])} 至 {np.max(groundwater[:,1])} m")
    
    # 创建深度网格 (假设简单模型: 0-40km深度)
    depth_grid = np.linspace(0, 40e3, grid_lon.shape[0])  # 40km深度
    depth_grid = np.tile(depth_grid, (grid_lon.shape[1], 1)).T  # 扩展为二维网格
    
    return rbf_ve,rbf_vn, {
        'lons': grid_lon,
        'lats': grid_lat,
        'x': grid_x,
        'y': grid_y,
        'dx': dx,
        'dy': dy,
        've': ve_grid,
        'vn': vn_grid,
        'dilatation': dilatation_grid,
        'max_shear': shear_strain_grid,
        'rotation': rotation_grid,
        'groundwater': gw_grid if gw_grid is not None else np.zeros_like(grid_x),
        'depth': depth_grid  # 新增深度网格
    
    }

# 3. 计算应变率张量
def calculate_strain_tensor(data):
    """
    使用 Sobel 滤波器计算应变率张量，单位保留为 1/yr
    输入：
        data['ve'], data['vn']：东、北方向速度分量（单位 mm/yr）
        dx, dy：水平网格间距（单位：度或米，需与投影一致）
    输出：
        exx, eyy, exy：对角与剪切应变率张量（单位 1/yr）
    """
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import sobel
    print("→ 使用 Sobel 滤波计算应变率张量...")
    # 网格间距单位应为米（不除以1000）
    dx = data['dx']  # 单位：m
    dy = data['dy']  # 单位：m
    # 平滑原始速度数据（避免高频震荡）
    ve = gaussian_filter(data['ve'], sigma=1)
    vn = gaussian_filter(data['vn'], sigma=1)

    # mm/yr → m/yr（保持单位一致性）
    ve = ve / 1000
    vn = vn / 1000

    # Sobel 滤波器求导数（更平滑）
    dvx_dx = sobel(ve, axis=1, mode='nearest') / (8 * dx)
    dvy_dy = sobel(vn, axis=0, mode='nearest') / (8 * dy)
    dvx_dy = sobel(ve, axis=0, mode='nearest') / (8 * dy)
    dvy_dx = sobel(vn, axis=1, mode='nearest') / (8 * dx)

    # 构造应变率张量
    exx = dvx_dx   # ε_xx
    eyy = dvy_dy   # ε_yy
    exy = 0.5 * (dvx_dy + dvy_dx)  # 对称剪应变率张量
    # 使用膨胀率约束垂向分量（注意单位一致性）
    ezz = data['dilatation'] - (exx + eyy)  # yr⁻¹
    
    # 计算旋转率 (rad/yr)
    omega_calc = 0.5 * (dvy_dx - dvx_dy) / 1000.0  # rad/yr
    
    
    print(f"应变率范围: exx={np.nanmin(exx):.2e}-{np.nanmax(exx):.2e} yr⁻¹")
    print(f"计算旋转率范围: {np.nanmin(omega_calc):.2e}-{np.nanmax(omega_calc):.2e} rad/yr")
    print(f"观测旋转率范围: {np.nanmin(data['rotation']):.2e}-{np.nanmax(data['rotation']):.2e} rad/yr")
    sample_idx = 100  # 示例点
    print(f"示例点验证:")
    print(f"  dvx_dx = {dvx_dx.flat[sample_idx]:.2e} mm/yr/m")
    print(f"  exx = {exx.flat[sample_idx]:.2e} yr⁻¹")
    print(f"  omega_calc = {omega_calc.flat[sample_idx]:.2e} rad/yr")
    print(f"  omega_obs = {data['rotation'].flat[sample_idx]:.2e} rad/yr")
    return exx, eyy, ezz, exy

def get_viscosity_by_depth(depth):
    """根据深度返回对应的黏滞系数"""
    for depth_boundary, viscosity in reversed(VISCOSITY_LAYERS):
        if depth >= depth_boundary:
            return viscosity
    return VISCOSITY_LAYERS[-1][1]  # 默认返回最底层的值
# 4. 计算应力率张量
def calculate_stress_rate_tensor(exx, eyy, ezz, exy, depth_grid=None, default_eta=1e21):
    """Maxwell黏弹性模型下的应力率张量计算,，支持分层黏滞系数"""
    print("使用 Maxwell 黏弹性模型计算应力率张量...")
    # 如果没有提供深度网格，使用默认黏滞系数
    if depth_grid is None:
        eta_grid = np.full_like(exx, default_eta)
    else:
        # 为每个网格点计算对应的黏滞系数
        eta_grid = np.vectorize(get_viscosity_by_depth)(depth_grid)
    # 拉梅常数
    lam = LAMBDA

    # 黏度η (Pa·s)，默认为1e19 Pa·s
    # 时间单位：年 → 秒
    year_sec = 365.25 * 24 * 3600
    
    # 应变率单位为 1/年，需转换为 1/秒
    exx_s = exx / year_sec
    eyy_s = eyy / year_sec
    ezz_s = ezz / year_sec
    exy_s = exy / year_sec
    # 应力率分量计算（Pa/s）
    tr = exx_s + eyy_s + ezz_s
    # 使用网格化的黏滞系数计算松弛因子
    relaxation_factor = 1 / (1 + G / eta_grid)
    
    sxx_dot = (2 * G * exx_s + lam * tr)* relaxation_factor
    syy_dot = (2 * G * eyy_s + lam * tr)* relaxation_factor
    szz_dot = (2 * G * ezz_s + lam * tr)* relaxation_factor
    sxy_dot = (2 * G * exy_s)* relaxation_factor
    
    sxz_dot = np.zeros_like(sxx_dot)
    syz_dot = np.zeros_like(sxx_dot)

    # 可加入应力松弛项 -σ/η，如果已有历史应力张量的话

    # 转换为 Pa/yr
    sxx = sxx_dot * year_sec
    syy = syy_dot * year_sec
    szz = szz_dot * year_sec
    sxy = sxy_dot * year_sec
    

    return sxx, syy, szz, sxy, sxz_dot, syz_dot


# 5. 计算库仑应力率
def calculate_coulomb_stress_rate(sxx, syy, szz, sxy, sxz, syz, data):
    """
    计算库仑应力率(CSR)
    
    """
    print("计算库仑应力率...")
    # 加入区域构造约束
    # regional_strike = np.deg2rad(85)  # 小江断裂走向85°
    # theta = 0.7*theta + 0.3*regional_strike  # 70%数据驱动 + 30%先验知识
    
    # 固定断层几何（巧家-东川段，小江断裂）
    strike_deg = 167
    dip_deg = 85
    rake_deg = -10
    # 转换为弧度
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    rake = np.deg2rad(rake_deg)
    # 断层法向量
    nx =-np.sin(dip) * np.sin(strike) 
    ny = np.sin(dip) * np.cos(strike)
    nz = -np.cos(dip)
    normal = np.array([nx, ny, nz])
    
    # 滑动方向（slip vector）
    sx =  np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike)
    sy =  np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike)
    sz =  np.sin(rake) * np.sin(dip)
    slip = np.array([sx, sy, sz])
    # 初始化数组
    csr_grid = np.zeros_like(sxx)
    sigma_n = np.zeros_like(sxx)
    tau = np.zeros_like(sxx)
    
    # 计算孔隙压变化 (Pa)
    # 孔隙压变化率 (考虑空间衰减)
    delta_Pp_rate = RHO_WATER * GRAVITY * data['groundwater']    # 水位变化率m/yr，孔隙压变化率 ρ·g·dh/dt (Pa/yr)
    for i in range(sxx.shape[0]):
        for j in range(sxx.shape[1]):
            stress_tensor = np.array([
                [sxx[i,j], sxy[i,j], sxz[i,j]],
                [sxy[i,j], syy[i,j], syz[i,j]],
                [sxz[i,j], syz[i,j], szz[i,j]]
            ])
            
            
            # 保持数组结构的关键修改
            sigma_n[i,j] = normal @ stress_tensor @ normal
            tau[i,j] = slip @ stress_tensor @ normal
            # 添加孔隙压效应: ΔCFS = Δτ + μ'(Δσ_n + ΔP_p)
            csr_grid[i,j] = tau[i,j] + MU_PRIME *(sigma_n[i,j] + delta_Pp_rate[i,j])
    print(f"断层走向: {strike:.1f}°")
    # 抽样检查（现在sigma_n是正确数组）
    print("正应力统计:")
    print(f"均值: {np.nanmean(sigma_n):.2f} Pa/yr")
    print(f"标准差: {np.nanstd(sigma_n):.2f} Pa/yr")
    # print(f"孔隙压: {delta_Pp_rate:.2f} Pa/yr")
    sec_per_year = 365.25 * 24 * 3600
   # 最终输出单位转换为kPa/yr
    csr_pa_per_yr = csr_grid  # 已经是Pa/yr
    csr_kpa_per_yr = csr_pa_per_yr / 1000.0  # kPa/yr
    from scipy.ndimage import gaussian_filter

    csr_kpa_per_yr = gaussian_filter(csr_kpa_per_yr, sigma=1)  # sigma 控制平滑程度

    print(f"库仑应力率范围: {np.nanmin(csr_kpa_per_yr):.2f}-{np.nanmax(csr_kpa_per_yr):.2f} kPa/yr")
    return csr_kpa_per_yr


# 读取小江断裂带所有段
def read_fault_segments(text_lines):
    """
    将轨迹文本行分类组织成不同断层段
    返回：{ 'Xiaojiang_Fault_01': [(lon, lat), ...], ... }
    """
    from collections import defaultdict
    segment_dict = defaultdict(list)
    for line in text_lines:
        parts = line.strip().split()
        name = parts[0]
        lon = float(parts[1])
        lat = float(parts[2])
        segment_dict[name].append((lon, lat))
    return segment_dict
# 构造巧家-东川段的断层轨迹带
def build_fault_band(fault_points, half_width_km=2):
    """
    根据断层轨迹点生成带宽为 ±half_width_km 的断裂带多边形点串
    """
    geod = Geod(ellps='WGS84')
    left_band = []
    right_band = []
    fault_points = np.array(fault_points) 
    for i in range(len(fault_points)-1):
        lon1, lat1 = fault_points[i]
        lon2, lat2 = fault_points[i+1]

        az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)
        left_az = az12 - 90
        right_az = az12 + 90

        # 向左、右偏移一定距离
        lon_l1, lat_l1, _ = geod.fwd(lon1, lat1, left_az, half_width_km * 1000)
        lon_r1, lat_r1, _ = geod.fwd(lon1, lat1, right_az, half_width_km * 1000)

        lon_l2, lat_l2, _ = geod.fwd(lon2, lat2, left_az, half_width_km * 1000)
        lon_r2, lat_r2, _ = geod.fwd(lon2, lat2, right_az, half_width_km * 1000)

        left_band.append((lon_l1, lat_l1))
        right_band.append((lon_r1, lat_r1))
        if i == len(fault_points)-2:
            left_band.append((lon_l2, lat_l2))
            right_band.append((lon_r2, lat_r2))

    # 构建多边形封闭区域点串：左侧点 + 右侧点反向
    band_polygon = left_band + right_band[::-1]
    return band_polygon


# 6. 计算巧家-东川段的库伦应力
def compute_cfs_on_fault_segment(
    fault_points,
    sxx, syy, szz, sxy, sxz, syz,
    grid_lat, grid_lon,
    strike_deg=167, dip_deg=85, rake_deg=-10, 
    mu_prime=0.4
):
    """
    计算给定断层轨迹点上的库仑应力变化 (ΔCFS)，单位 kPa/yr
    """
    # 区域网格坐标
    
    
    # 将断层点格式统一为 numpy 数组
    fault_points = np.array(fault_points)  # shape: (N, 2)
    # print(fault_points)
    # print(grid_lat,grid_lon)
    # 插值器（注意网格纬度、经度顺序）
    interp_sxx = RegularGridInterpolator((grid_lat, grid_lon), sxx,bounds_error=False, fill_value=np.nan)
    interp_syy = RegularGridInterpolator((grid_lat, grid_lon), syy,bounds_error=False, fill_value=np.nan)
    interp_szz = RegularGridInterpolator((grid_lat, grid_lon), szz,bounds_error=False, fill_value=np.nan)
    interp_sxy = RegularGridInterpolator((grid_lat, grid_lon), sxy,bounds_error=False, fill_value=np.nan)
    interp_sxz = RegularGridInterpolator((grid_lat, grid_lon), sxz,bounds_error=False, fill_value=np.nan)
    interp_syz = RegularGridInterpolator((grid_lat, grid_lon), syz,bounds_error=False, fill_value=np.nan)

    # 插值点（注意：lat, lon）
    points = np.flip(fault_points, axis=1)
    # print(points)
    sxx_f = interp_sxx(points)
    syy_f = interp_syy(points)
    szz_f = interp_szz(points)
    sxy_f = interp_sxy(points)
    sxz_f = interp_sxz(points)
    syz_f = interp_syz(points)

    # 转换断层几何为弧度
    strike = np.deg2rad(strike_deg)
    dip = np.deg2rad(dip_deg)
    rake = np.deg2rad(rake_deg)

    # 断层法向量
    n = np.array([
        -np.sin(dip) * np.sin(strike),
         np.sin(dip) * np.cos(strike),
        -np.cos(dip)
    ])

    # 滑移方向向量
    s = np.array([
        np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
        np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
        np.sin(rake) * np.sin(dip)
    ])

    # 计算 ΔCFS
    cfs_list = []
    for i in range(len(points)):
        sigma = np.array([
            [sxx_f[i], sxy_f[i], sxz_f[i]],
            [sxy_f[i], syy_f[i], syz_f[i]],
            [sxz_f[i], syz_f[i], szz_f[i]],
        ])
        sigma_n = n @ sigma @ n
        tau = s @ sigma @ n
        cfs = tau + mu_prime * sigma_n
        cfs_list.append(cfs)
    cfs_values = np.array(cfs_list)/ 1000.0
    # 绘图
    # fig = plt.figure(figsize=(10,8))
    # ax = plt.axes(projection=ccrs.PlateCarree())

    # # 添加地图要素
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.set_extent([102.8, 103.5, 25.6, 26.9])

    # # 绘制断裂带面
    # poly = Polygon(band_polygon, facecolor='none', edgecolor='black', linewidth=1.5)
    # ax.add_patch(poly)

    # # 绘制彩色圆点表示 ΔCFS
    # sc = ax.scatter(
    #     [pt[0] for pt in fault_points],
    #     [pt[1] for pt in fault_points],
    #     c=cfs_values,
    #     cmap='seismic',
    #     s=80,
    #     edgecolor='k',
    #     transform=ccrs.PlateCarree()
    # )

    # plt.colorbar(sc, ax=ax, label='ΔCFS (kPa/yr)')
    # plt.title("巧家–东川段断裂带上的库仑应力变化")
    # plt.tight_layout()
    # plt.show()
    # 返回 ΔCFS（单位 kPa/yr）
    return cfs_values

# 7.计算小江断裂所有段的库仑应力
def compute_all_segments_cfs(
    segment_dict,  # 字典结构，每段的点
    grid_lon, grid_lat, sxx, syy,szz, sxy,sxz,syz,
    strike_deg=167, dip_deg=85, rake_deg=-10,
    mu=0.4,
    half_width_km=2
):
    """
    返回：列表，每个元素是 (name, coords, ΔCFS, 带宽多边形)
    """
    from collections import namedtuple
    Result = namedtuple('Result', ['name', 'coords', 'cfs', 'band'])
    grid_x = grid_lon[0, :]
    grid_y = grid_lat[:, 0]
    results = []
    # print(segment_dict)
    for name, points in segment_dict.items():
        if len(points) < 2:
            continue  # 轨迹太短跳过
        # 计算 ΔCFS
        cfs_vals = compute_cfs_on_fault_segment(
            points,
            sxx, syy, szz, sxy, sxz, syz,
            grid_y, grid_x
        )

        # 构建带宽面
        band_polygon = build_fault_band(points, half_width_km=half_width_km)

        results.append(Result(name, points, cfs_vals, band_polygon))
    
    return results

def solve_poisson(rhs, dx, dy):
    """
    求解二维泊松方程 ∇²φ = rhs，返回 φ
    Dirichlet 边界条件：边界值设为 0
    """
    from scipy.ndimage import gaussian_filter
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import Rbf, RegularGridInterpolator
    ny, nx = rhs.shape
    N = nx * ny
    dx2 = dx ** 2
    dy2 = dy ** 2
    A = diags([
        -1 / dy2 * np.ones(N - nx),    # 下
        -1 / dx2 * np.ones(N - 1),     # 左
        2 * (1 / dx2 + 1 / dy2) * np.ones(N),  # 中
        -1 / dx2 * np.ones(N - 1),     # 右
        -1 / dy2 * np.ones(N - nx)     # 上
    ], [-nx, -1, 0, 1, nx], shape=(N, N), format='csr')

    rhs_flat = rhs.flatten()
    phi = spsolve(A, rhs_flat)
    return phi.reshape((ny, nx))

def integrate_velocity_from_strain(exx, eyy, exy, dx, dy,grid_data, gnss_data):
    """
    采用泊松方程方法从应变率恢复速度场
    输出单位：mm/yr
    """
    print("→ 使用泊松方程+ GNSS速度约束恢复速度场求解速度场...")

    # 将应变率单位从 1/yr 转为 1/s（匹配应力单位），但速度单位最终保留 mm/yr
    # 不需要额外单位变换
    from scipy.ndimage import gaussian_filter
    exx = gaussian_filter(exx, sigma=1)
    eyy = gaussian_filter(eyy, sigma=1)
    exy = gaussian_filter(exy, sigma=1)
    # 计算右端项 RHS
    d_exx_dx = np.gradient(exx, dx, axis=1)
    d_exy_dy = np.gradient(exy, dy, axis=0)
    rhs_ve = d_exx_dx + d_exy_dy

    d_exy_dx = np.gradient(exy, dx, axis=1)
    d_eyy_dy = np.gradient(eyy, dy, axis=0)
    rhs_vn = d_exy_dx + d_eyy_dy

    # 解泊松方程（单位 m/yr → mm/yr）
    ve_pure = solve_poisson(rhs_ve, dx, dy) * 1000
    vn_pure = solve_poisson(rhs_vn, dx, dy) * 1000

    # -----------------------------
    # GNSS点约束速度场构建
    # -----------------------------
    print("→ 插值GNSS速度点为目标约束...")

    interp_ve_target = RegularGridInterpolator(
        (grid_data['lats'][:, 0], grid_data['lons'][0, :]), ve_pure, bounds_error=False, fill_value=None
    )
    interp_vn_target = RegularGridInterpolator(
        (grid_data['lats'][:, 0], grid_data['lons'][0, :]), vn_pure, bounds_error=False, fill_value=None
    )

    lat_pts = gnss_data[:, 1]
    lon_pts = gnss_data[:, 0]
    points = np.column_stack((lat_pts, lon_pts))

    ve_simulated_at_gnss = interp_ve_target(points)
    vn_simulated_at_gnss = interp_vn_target(points)

    ve_gnss_obs = gnss_data[:, 2]
    vn_gnss_obs = gnss_data[:, 3]

    ve_residual = ve_gnss_obs - ve_simulated_at_gnss
    vn_residual = vn_gnss_obs - vn_simulated_at_gnss

    print(f"→ 残差速度范围: ve={ve_residual.min():.2f}~{ve_residual.max():.2f}, vn={vn_residual.min():.2f}~{vn_residual.max():.2f}")

    # -----------------------------
    # RBF 插值残差修正场（用于锚定）
    # -----------------------------
    print("→ 使用RBF拟合残差场进行全场修正...")

    rbf_ve_corr = Rbf(lon_pts, lat_pts, ve_residual, function='multiquadric', smooth=0.5)
    rbf_vn_corr = Rbf(lon_pts, lat_pts, vn_residual, function='multiquadric', smooth=0.5)

    ve_corr_grid = rbf_ve_corr(grid_data['lons'], grid_data['lats'])
    vn_corr_grid = rbf_vn_corr(grid_data['lons'], grid_data['lats'])

    # -----------------------------
    # 最终速度场 = 泊松速度场 + 修正场
    # -----------------------------
    ve_final = ve_pure + ve_corr_grid
    vn_final = vn_pure + vn_corr_grid

    return ve_final, vn_final


def generate_background_velocity_from_strain(data,exx, eyy,exy):
    """
    利用剪切率和主剪切角构造背景速度场（单位 mm/yr）
    """

    print("→ 构建背景速度场（方向依据主剪切轴）...")

    # 1. 计算剪切率（绝对值）
    shear_rate = np.sqrt((exx - eyy) ** 2 + 4 * exy ** 2) / 2  # 单位 1/yr

    # 2. 计算主剪切方向
    theta = theta = 0.5 * np.arctan2(2 * exy, exx - eyy)
    print(f"主应力方向：{theta}")
    dx = data['dx']  # 单位：m
    dy = data['dy']  # 单位：m
    
    # 3. 构造背景速度分量（假设总速度 = 局部剪切率 × 空间尺度 × 方向）
    scale = 0.5 * (dx + dy)  # 代表局部剪切影响的长度尺度（单位：m）
    ve_bg = shear_rate * scale * np.cos(theta) * 1000  # mm/yr
    vn_bg = shear_rate * scale * np.sin(theta) * 1000  # mm/yr

    return ve_bg, vn_bg

def interpolate_velocity_to_gnss(ve_field, vn_field, grid_lons, grid_lats, gnss_data):
    """
    插值模拟速度场至 GNSS 点
    """
    interp_ve = RegularGridInterpolator((grid_lats[:, 0], grid_lons[0, :]), ve_field)
    interp_vn = RegularGridInterpolator((grid_lats[:, 0], grid_lons[0, :]), vn_field)
    
    points = np.column_stack((gnss_data[:, 1], gnss_data[:, 0]))  # lat, lon
    ve_model = interp_ve(points)
    vn_model = interp_vn(points)
    
    return ve_model, vn_model
def plot_gnss_comparison_quiver(gnss_data, ve_model, vn_model, scale=1/0.01):
    """
    绘制模拟 vs 观测 GNSS 速度场对比图
    """
    fig, axs = plt.subplots(figsize=(8, 8))
    plt.grid(True)
    # plt.tight_layout()
    lons = gnss_data[:, 0]
    lats = gnss_data[:, 1]

    ve_obs = gnss_data[:, 2]
    vn_obs = gnss_data[:, 3]
    ve_res = ve_obs - ve_model
    vn_res = vn_obs - vn_model
    
    # 读取转换后的经纬度数据
    data_file = "f:/鲁人齐最新全国断层数据/附表1-2与断层数据/processed_fault_data.txt"  # 你的数据文件路径
    data = pd.read_csv(data_file, delim_whitespace=True, header=None, 
                    names=[ "FaultName", "Longitude", "Latitude"])
    # 获取所有唯一的 Faultname
    data = data[
        (data["Longitude"] >= LON_MIN) & (data["Longitude"] <= LON_MAX) & 
        (data["Latitude"] >= LAT_MIN) & (data["Latitude"] <= LAT_MAX)
    ]
    fault_names = data['FaultName'].unique()

    for fault in fault_names:
        # 筛选出当前 Faultname 的数据
        
        fault_data = data[data['FaultName'] == fault]
        # 按照 Y（纬度）列排序，ascending=True 表示从小到大排序
        # fault_data = fault_data.sort_values(by='Latitude').reset_index(drop=True)  
        # print(fault_data)

        axs.plot(fault_data["Longitude"], fault_data["Latitude"],  color='lightgray', lw=1.5, linestyle='-')
    # ==== 左图：观测 vs 模拟 ====
    axs.quiver(lons, lats, ve_obs, vn_obs, color='red', scale=scale, label='observed')
    axs.quiver(lons, lats, ve_model, vn_model, color='blue',alpha=0.8, scale=scale, label='simulated')
    axs.set_title("GNSS comparison", fontsize=14)
    axs.set_xlabel("Longitude")
    axs.set_ylabel("Latitude")
    # axs.set_ylim(LAT_MIN-0.1, LAT_MAX)
    
    axs.set_aspect('equal')
    axs.legend()
    
    
    
    plt.savefig('gnss_velocity_comparison.svg', dpi=650)
    plt.show()

def plot_residual_heatmap(gnss_data, residuals):
    """
    绘制速度残差热图
    """
    lon = gnss_data[:, 0]
    lat = gnss_data[:, 1]

    grid_lon = np.linspace(np.min(lon), np.max(lon), 100)
    grid_lat = np.linspace(np.min(lat), np.max(lat), 100)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    grid_res = griddata((lon, lat), residuals, (grid_x, grid_y), method='linear')

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_res, cmap='hot_r')
    plt.colorbar(label='residual (mm/yr)')
    plt.title('GNSS residual heatmap', fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.scatter(lon, lat, c='k', s=5, label='GNSS station')
    plt.legend()
    plt.savefig('gnss_residual hotmap.svg', dpi=650)
    plt.show()

#画gnss模拟对比图
def evaluate_velocity_model(gnss_data, exx, eyy, exy, grid_data):
    """
    综合评估速度场模拟效果（方法3+方法4）
    """
    print("→ 从应变率场恢复模拟速度场...")
    ve_sim, vn_sim = integrate_velocity_from_strain(exx, eyy, exy, grid_data['dx'], grid_data['dy'],grid_data, gnss_data)
    print("→ 添加构造背景速度场...")
    # origin = [np.mean(grid_data['lons']), np.mean(grid_data['lats'])]
    ve_bg, vn_bg = generate_background_velocity_from_strain(grid_data, exx,eyy, exy)
    ve_total = ve_sim + ve_bg
    vn_total = vn_sim + vn_bg
    
    print("→ 插值模拟速度到 GNSS 观测点...")
    ve_model, vn_model = interpolate_velocity_to_gnss(
        ve_total, vn_total, grid_data['lons'], grid_data['lats'], gnss_data
    )

    print("→ 计算残差与 RMSE...")
    ve_obs = gnss_data[:, 2]
    vn_obs = gnss_data[:, 3]
    ve_diff = ve_obs - ve_model
    vn_diff = vn_obs - vn_model
    residuals = np.sqrt(ve_diff**2 + vn_diff**2)

    rms_ve = np.sqrt(np.mean(ve_diff**2))
    rms_vn = np.sqrt(np.mean(vn_diff**2))
    rms_total = np.sqrt(np.mean(residuals**2))
    
    angle_diff = np.arctan2(vn_obs, ve_obs) - np.arctan2(vn_model, ve_model)
    angle_diff_deg = np.rad2deg(angle_diff)
    print(f"角度差:{angle_diff_deg}")
    print(f"→ 平均 RMS 误差：")
    print(f"   东向 RMS: {rms_ve:.2f} mm/yr")
    print(f"   北向 RMS: {rms_vn:.2f} mm/yr")
    print(f"   合速度 RMS: {rms_total:.2f} mm/yr")


    print("→ 绘图：矢量场对比...")
    plot_gnss_comparison_quiver(gnss_data, ve_model, vn_model)

    print("→ 绘图：速度残差热图...")
    plot_residual_heatmap(gnss_data, residuals)
    
# 画gnss边界图
def plot_velocity_field_with_boundaries(ax, ve_func, vn_func, bounds,
                                        obs_data=None, scale=0.1, arrow_color='#297270'):
    """
    绘制边界和区域内的 GNSS 模拟速度场箭头（Rbf 插值版本）。
    """
    min_lon, max_lon, min_lat, max_lat = bounds
    # 读取转换后的经纬度数据
    data_file = "f:/鲁人齐最新全国断层数据/附表1-2与断层数据/processed_fault_data.txt"  # 你的数据文件路径
    data = pd.read_csv(data_file, delim_whitespace=True, header=None, 
                    names=[ "FaultName", "Longitude", "Latitude"])

    data = data[
        (data["Longitude"] >= LON_MIN) & (data["Longitude"] <= LON_MAX) & 
        (data["Latitude"] >= LAT_MIN) & (data["Latitude"] <= LAT_MAX)
    ]
    fault_names = data['FaultName'].unique()

    for fault in fault_names:
        # 筛选出当前 Faultname 的数据
        
        fault_data = data[data['FaultName'] == fault]
        # 按照 Y（纬度）列排序，ascending=True 表示从小到大排序
        # fault_data = fault_data.sort_values(by='Latitude').reset_index(drop=True)  
        # print(fault_data)

        ax.plot(fault_data["Longitude"], fault_data["Latitude"],  color='lightgray', lw=1.5, linestyle='-')
    # 2. 观测 GNSS（可选）
    if obs_data is not None:
        ax.quiver(obs_data[:,0], obs_data[:,1], obs_data[:,2], obs_data[:,3],
                  color='#e66d50', scale=scale,  label='observed GNSS')
        
    # 3. 四周边界箭头
    edge_N = np.linspace(min_lon, max_lon, 20)
    edge_S = edge_N.copy()
    edge_E = np.linspace(min_lat, max_lat, 20)
    edge_W = edge_E.copy()

    # 北边界
    x_n,y_n = project_coords(edge_N, [max_lat]*len(edge_N))
    vx_n = ve_func(x_n, y_n)
    vy_n = vn_func(x_n, y_n) 
    ax.quiver(edge_N, [max_lat]*20, vx_n, vy_n, color=arrow_color, scale=scale,label='boundary constraints')

    # 南边界
    x_s, y_s = project_coords(edge_S, [min_lat]*len(edge_S))
    vx_s = ve_func(x_s, y_s)
    vy_s = vn_func(x_s, y_s) 
    ax.quiver(edge_S, [min_lat]*20, vx_s, vy_s, color=arrow_color, scale=scale)

    # 西边界
    x_w, y_w = project_coords([min_lon]*len(edge_W), edge_W)
    vx_w = ve_func(x_w, y_w)
    vy_w = vn_func(x_w, y_w) 
    ax.quiver([min_lon]*20, edge_W, vx_w, vy_w, color=arrow_color, scale=scale)

    # 东边界
    x_e, y_e = project_coords([max_lon]*len(edge_E), edge_E)
    vx_e = ve_func(x_e, y_e)
    vy_e = vn_func(x_e, y_e) 
    ax.quiver([max_lon]*20, edge_E, vx_e, vy_e, color=arrow_color, scale=scale)

    
    # # 2. 区域内部箭头（模拟 GNSS）
    # lon_grid = np.linspace(min_lon, max_lon, 30)
    # lat_grid = np.linspace(min_lat, max_lat, 30)
    # lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    # x_mesh,y_mesh = project_coords(lon_mesh, lat_mesh)
    # vx_vals = ve_func(x_mesh, y_mesh)
    # vy_vals = vn_func(x_mesh, y_mesh)

    # ax.quiver(lon_mesh, lat_mesh, vx_vals, vy_vals,
    #           color='black', scale=scale, width=0.002, label='模拟速度')
    
    
    
    # 4. 比例尺箭头
    scale_lon = min_lon + 0.2
    scale_lat = min_lat + 0.2
    ax.quiver(scale_lon, scale_lat, 20, 0,
              color='black', scale=scale, width=0.005)
    ax.text(scale_lon + 0.1, scale_lat + 0.05, '20 mm/yr', fontsize=8)

    ax.legend(fontsize=8, loc='upper right')

# 画小江断裂带上的库仑应力
def plot_all_fault_segments(results, extent=None):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Polygon
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='lightblue', alpha=0.4)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='blue', alpha=0.3)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='--', edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5)

    if extent:
        ax.set_extent(extent)
        
    # 添加经纬度网格和标签
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True,  # 启用标签
        linewidth=1, 
        color='gray', 
        alpha=0.5,
        linestyle='--'
    )
    # 配置标签位置：底部和左侧显示标签
    gl.top_labels = False    # 顶部不显示标签
    gl.right_labels = False  # 右侧不显示标签
    gl.bottom_labels = True  # 底部显示经度标签
    gl.left_labels = True    # 左侧显示纬度标签
    # 设置标签格式

    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
        
    # all_cfs = np.concatenate([r.cfs for r in results])
    norm = mcolors.Normalize(vmin=-5, vmax=10)
    cmap = plt.get_cmap("seismic")  # 黄色到红色，强调“危险高值”

    
    for r in results:
        # 面
        # 取该段的 ΔCFS 平均值作为颜色
        max_cfs = np.nanmax(r.cfs)
        print(f"{r.name} 的最大 ΔCFS = {max_cfs:.2f} kPa/yr")
        coords = r.coords
        cfs = r.cfs
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i+1]
            cfs_val = np.nanmean([cfs[i], cfs[i+1]])

            # 构造该段的 polygon 面
            band = build_fault_band([p1, p2], half_width_km=2)

            # 着色
            color = cmap(norm(cfs_val))

            ax.add_patch(Polygon(band, facecolor=color, edgecolor='none'))
        # 彩色圆点
        # sc = ax.scatter(
        #     [pt[0] for pt in r.coords],
        #     [pt[1] for pt in r.coords],
        #     c=r.cfs,
        #     cmap='seismic',
        #     # vmin=-cmax, vmax=cmax,
        #     s=80,
        #     edgecolor='k',
        #     transform=ccrs.PlateCarree(),
        #     label=r.name
        # )

    # 创建 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='ΔCFS (kPa/yr)')
    # 城市名称与坐标（示例，可自行扩展）
    cities = {
        'Kunming': (25.04, 102.72),
        'Dongchuan': (26.08, 103.19),
        'Qiaojia': (26.90, 102.93),
        'Zhaotong': (27.33, 103.72)
        
       
    }

    for name, (lat, lon) in cities.items():
        ax.scatter(lon, lat, c='black', s=10, zorder=10)
        # ax.text(lon + 0.05, lat + 0.05, name, fontsize=8, ha='left', va='bottom', color='black')
    # plt.title(title='小江断裂带各段 ΔCFS')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig('cfs_segments_all.svg', dpi=650)
    plt.show()
    
    

# 新增：改进的Rate-State模型触发概率计算
def rate_state_probability(dCFS, tau, A_sigma, t_elapsed, b=0.01):
    """
    Dieterich (1994) Rate-State 触发概率模型
    基于原文方程 (11)(12)(14)

    参数:
    dCFS      : 库仑应力变化 ΔCFS，对应原文 Δτ (Pa)
    tau       : 弛豫时间 t_a = Aσ / τ̇_r (年)，对应原文 t_a
    A_sigma   : 状态依赖参数 Aσ (Pa)，对应原文 Aσ
    t_elapsed : 应力阶跃后经过的时间 t (年)
    b         : 背景地震率 r (events/yr)，对应原文 r

    返回:
    P         : 泊松过程触发概率 [0, 1]
    """
    # 原文方程 (12): R(t) = r / { [exp(-ΔCFS/Aσ) - 1] · exp(-t/t_a) + 1 }
    exp_neg   = np.exp(-dCFS / A_sigma)          # exp(-ΔCFS / Aσ)
    decay     = np.exp(-t_elapsed / tau)          # exp(-t / t_a)
    R         = b / ((exp_neg - 1.0) * decay + 1.0)

    # 对 R(t) 在 [0, t_elapsed] 上的解析积分:
    # ∫₀ᵀ R(t)dt = r · t_a · ln[ (exp(T/t_a) - 1 + exp(ΔCFS/Aσ)) / (exp(ΔCFS/Aσ) - 1) ]
    exp_pos   = np.exp( dCFS / A_sigma)           # exp(+ΔCFS / Aσ)
    exp_T     = np.exp( t_elapsed / tau)          # exp(T / t_a)
    integral  = b * tau * np.log(
        (exp_T - 1.0 + exp_pos) / (exp_pos - 1.0)
    )

    # 泊松过程概率转换: P = 1 - exp(-∫R dt)
    return 1.0 - np.exp(-integral)


# 8. 验证地震目录可靠性
def validate_catalog(earthquakes, csr, grid_lons, grid_lats, groundwater):
    """验证地震目录与CSR的相关性"""
    # 插值地震位置的CSR值
    print("验证地震目录...")
    # 投影地震坐标
    eq_lons, eq_lats = earthquakes[:, 0], earthquakes[:, 1]
    

    # 插值地震位置的CSR值
    csr_values = griddata(
        (grid_lons.flatten(), grid_lats.flatten()),
        csr.flatten(),
        (eq_lons, eq_lats),
        method='linear'
    )
    
    # 插值地震位置的地下水位变化值
    gw_values = np.zeros(len(eq_lons))
    if groundwater is not None:
        gw_values = griddata(
            (grid_lons.flatten(), grid_lats.flatten()),
            groundwater.flatten(),
            (eq_lons, eq_lats),
            method='linear'
        )
    # 统计检验
    valid_mask = np.isfinite(csr_values)
    # print(f"mask: {valid_mask}")
    n_total = np.sum(valid_mask)
    
    if n_total == 0:
        print("警告: 无有效地震可用于验证")
        return 0, 1, np.zeros_like(earthquakes[:,0])
    positive_mask = csr_values[valid_mask] >= 0
    n_pos = np.sum(positive_mask)
    ratio = n_pos / n_total
    
    # 卡方检验
    try:
        _, p, _, _ = chi2_contingency([[n_pos, n_total - n_pos]])
    except:
        p = 1.0
    
    print(f"\n验证结果:")
    print(f"有效地震数: {n_total}/{len(earthquakes)}")
    print(f"位于CSR>0区域的地震比例: {ratio:.1%} ")
    
    if p < 0.05 and ratio > 0.6:
        print("结论: 目录可靠性高 (显著聚集于应力积累区)")
    elif p < 0.05 and ratio < 0.4:
        print("警告: 目录可能存在问题 (偏离应力积累区)")
    else:
        print("提示: 目录与应力场无显著关联")
    
    # 计算触发概率
    print("计算地震触发概率...")
    prob_values = np.zeros(len(csr_values))
    
    # 1. 定义参数扫描范围
    # A_sigma: 0.05 MPa 到 0.6 MPa (稍微细化一点，画出来曲线更圆滑)
    asigma_range = np.linspace(0.05, 0.6, 25) # 单位 MPa
    
    # Tau: 典型的复发周期
    tau_levels = [5, 10, 20, 30, 50, 100] # 年
    
    # 固定离逝时间
    t_elapsed_fixed = 10.0 
    
    # 基准参数 (正文中使用的参数)
    baseline_asigma = 0.3
    baseline_tau = 10

    # 准备数据
    data_list = []
    
    dCFS_pa = csr_values * 1000.0 
    valid_mask = np.isfinite(dCFS_pa)
    dCFS_clean = dCFS_pa[valid_mask]
    
    # 确保水位数据形状匹配
    gw_clean = None
    if groundwater is not None:
        # 注意：这里必须用 valid_mask 筛选，否则长度不一致会报错
        gw_values = griddata(
            (grid_lons.flatten(), grid_lats.flatten()),
            groundwater.flatten(),
            (earthquakes[:, 0], earthquakes[:, 1]),
            method='linear'
        )
        gw_clean = gw_values[valid_mask]
    # -----------------------------------------------------------

    # 1. 定义参数
    asigma_range = np.linspace(0.05, 0.6, 25) 
    tau_levels = [5, 10, 20, 30, 50, 100] 
    t_elapsed_fixed = 10.0 
    
    # 基准参数
    baseline_asigma = 0.3
    baseline_tau = 10

    # 2. 循环计算敏感性曲线数据
    data_list = []
    
    for tau in tau_levels:
        for asigma_mpa in asigma_range:
            asigma_pa = asigma_mpa * 1e6
            
            # 计算概率
            probs_temp = rate_state_probability(dCFS_clean, tau, asigma_pa, t_elapsed_fixed)
            
            # 叠加水位权重
            if gw_clean is not None:
                gw_weight = 1 + 0.5 * np.abs(gw_clean)
                probs_temp = np.clip(probs_temp * gw_weight, 0, 1)

            mean_prob = np.mean(probs_temp)

            data_list.append({
                'A_sigma': asigma_mpa,
                'Tau': tau,
                'Mean Probability': mean_prob
            })
            # 【注意】删除了这里依赖循环判断 if tau == baseline... 的代码

    df = pd.DataFrame(data_list)

    # 3. 【核心修复】单独计算基准点 (Red Star) 的值
    # 这样无论 linspace 怎么变，红星一定能画出来
    print("单独计算基准点...")
    p_base_clean = rate_state_probability(dCFS_clean, baseline_tau, baseline_asigma*1e6, t_elapsed_fixed)
    
    if gw_clean is not None:
        w_base = 1 + 0.5 * np.abs(gw_clean)
        p_base_clean = np.clip(p_base_clean * w_base, 0, 1)
    
    # 获取基准点的均值用于画星号
    baseline_prob_val = np.mean(p_base_clean)
    print(f"基准点 (0.3 MPa, 10 yr) 的平均概率为: {baseline_prob_val:.4f}")

    # ==========================================
    # 4. 绘图
    # ==========================================
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("viridis", len(tau_levels))
    
    # 绘制曲线
    sns.lineplot(
        data=df, 
        x='A_sigma', 
        y='Mean Probability', 
        hue='Tau', 
        palette=palette,
        linewidth=2.5,
        marker='o',
        markersize=6,
        dashes=False
    )

    # 【核心修复】绘制基准点红星
    plt.scatter([baseline_asigma], [baseline_prob_val], 
                color='red', s=250, marker='*', zorder=100, # zorder设大一点，保证在最上层
                edgecolor='k', linewidth=1,
                label='Parameters in Main Text')
    
    # 图表装饰
    plt.title('Sensitivity of Mean Triggering Probability\nto Physical Parameters', fontsize=16, pad=15)
    plt.xlabel(r'Constitutive Parameter $A\sigma$ (MPa)', fontsize=14)
    plt.ylabel('Mean Triggering Probability', fontsize=14)
    plt.legend(title='Recurrence Interval $\\tau$ (yr)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_file = 'sensitivity_curves_final.svg'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 5. 【重要】准备返回值
    # ==========================================
    # 必须返回基准参数下的全场概率，用于后续画地图
    # 如果直接返回循环里的 prob_values，那是最后一轮循环(tau=100)的值，是错误的！
    
    prob_values_full = np.full_like(csr_values, np.nan)
    prob_values_full[valid_mask] = p_base_clean # 使用刚才单独计算的 p_base_clean

    return ratio, p, csr_values, prob_values_full



# 9. 可视化结果
def plot_results(data, csr, earthquakes, csr_values,prob_values, groundwater=None):
    """
    可视化库仑应力率分布、GNSS速度场和地震事件位置
    """
    print("绘制结果图像...")
    
    # 创建图形和子图布局
    fig = plt.figure(figsize=(18, 18))
    
    # 子图1: 库仑应力率分布
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    # ax1.add_feature(cfeature.BORDERS, linestyle=':')
    # ax1.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    gl1 = ax1.gridlines(draw_labels=True)
    gl1.top_labels = False    # 顶部不显示标签
    gl1.right_labels = False  # 右侧不显示标签
    gl1.bottom_labels = True  # 底部显示经度标签
    gl1.left_labels = True    # 左侧显示纬度标签
    # 库仑应力率图层（单位：kPa/yr）
    im1 = ax1.pcolormesh(
        data['lons'], data['lats'], csr,
        cmap='seismic', shading='auto', vmin=-10, vmax=10,
        transform=ccrs.PlateCarree()
    )
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.6)
    cbar1.set_label('CSR (kPa/yr)')
    # ax1.set_title('CSR distribution')
    
    jiaojihe_file = "f:/BHT/交际河断裂坐标.txt"  # 你的数据文件路径
    jiaojihe = pd.read_csv(jiaojihe_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(jiaojihe["Longitude"], jiaojihe["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(jiaojihe["Longitude"][5], jiaojihe["Latitude"][5], 'Jiaojihe', fontsize=8, ha='left', va='bottom', color='black')

    puduhe_file = "f:/BHT/普渡河断裂坐标.txt"  # 你的数据文件路径
    puduhe = pd.read_csv(puduhe_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(puduhe["Longitude"], puduhe["Latitude"],color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(puduhe["Longitude"][5], puduhe["Latitude"][5], 'Puduhe', fontsize=8, ha='left', va='bottom', color='black')

    xiaojiangdong_file = "f:/BHT/小江断裂东支坐标.txt"  # 你的数据文件路径
    xiaojiangdong = pd.read_csv(xiaojiangdong_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(xiaojiangdong["Longitude"], xiaojiangdong["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(xiaojiangdong["Longitude"][5], xiaojiangdong["Latitude"][5], 'Xiaojiangdong', fontsize=8, ha='left', va='bottom', color='black')
    xiaojiangxi_file = "f:/BHT/小江断裂西支坐标.txt"  # 你的数据文件路径
    xiaojiangxi = pd.read_csv(xiaojiangxi_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(xiaojiangxi["Longitude"], xiaojiangxi["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(xiaojiangxi["Longitude"][5], xiaojiangxi["Latitude"][5], 'Xiaojiangxi', fontsize=8, ha='left', va='bottom', color='black')

    zemuhe_file = "f:/BHT/则木河断裂坐标.txt"  # 你的数据文件路径
    zemuhe = pd.read_csv(zemuhe_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(zemuhe["Longitude"],zemuhe["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(zemuhe["Longitude"][5], zemuhe["Latitude"][5], 'Zemuhe', fontsize=8, ha='left', va='bottom', color='black')

    zhaoqiao_file = "f:/BHT/昭巧断裂坐标.txt"  # 你的数据文件路径
    zhaoqiao = pd.read_csv(zhaoqiao_file, delim_whitespace=True, header=None, 
                    names=["Longitude", "Latitude"])
    ax1.plot(zhaoqiao["Longitude"], zhaoqiao["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax1.text(zhaoqiao["Longitude"][5], zhaoqiao["Latitude"][5], 'Zhaoqiao', fontsize=8, ha='left', va='bottom', color='black')
    print("所有断层图已绘制完成。")
    
    # 绘制地震事件
    sc1 = ax1.scatter(
        earthquakes[:, 0], earthquakes[:, 1],
        c=csr_values, cmap='seismic', vmin=-5, vmax=5,
        s=5, edgecolors='white', linewidths=0.5, alpha=0.8,
        transform=ccrs.PlateCarree()
    )
    # cities = {
    #     'Kunming': (25.04, 102.72),
    #     'Dongchuan': (26.08, 103.19),
    #     'Qiaojia': (26.90, 102.93),
    #     'Zhaotong': (27.33, 103.72)
        
       
    # }

    # for name, (lat, lon) in cities.items():
    #     ax1.scatter(lon, lat, c='black', s=10, zorder=10)
    #     ax1.text(lon + 0.05, lat + 0.05, name, fontsize=8, ha='left', va='bottom', color='black')
    # plt.title(title='小江断裂带各段 ΔCFS')
    
    # 子图2: 触发概率分布
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    ax2.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    # ax2.add_feature(cfeature.BORDERS, linestyle=':')
    # ax2.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    gl2 = ax2.gridlines(draw_labels=True)
    gl2.top_labels = False    # 顶部不显示标签
    gl2.right_labels = False  # 右侧不显示标签
    gl2.bottom_labels = True  # 底部显示经度标签
    gl2.left_labels = True    # 左侧显示纬度标签
    
    ax2.plot(jiaojihe["Longitude"], jiaojihe["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(jiaojihe["Longitude"][5], jiaojihe["Latitude"][5], 'Jiaojihe', fontsize=8, ha='left', va='bottom', color='black')

    ax2.plot(puduhe["Longitude"], puduhe["Latitude"],color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(puduhe["Longitude"][5], puduhe["Latitude"][5], 'Puduhe', fontsize=8, ha='left', va='bottom', color='black')

    ax2.plot(xiaojiangdong["Longitude"], xiaojiangdong["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(xiaojiangdong["Longitude"][5], xiaojiangdong["Latitude"][5], 'Xiaojiangdong', fontsize=8, ha='left', va='bottom', color='black')

    ax2.plot(xiaojiangxi["Longitude"], xiaojiangxi["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(xiaojiangxi["Longitude"][5], xiaojiangxi["Latitude"][5], 'Xiaojiangxi', fontsize=8, ha='left', va='bottom', color='black')

    ax2.plot(zemuhe["Longitude"],zemuhe["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(zemuhe["Longitude"][5], zemuhe["Latitude"][5], 'Zemuhe', fontsize=8, ha='left', va='bottom', color='black')

    ax2.plot(zhaoqiao["Longitude"], zhaoqiao["Latitude"], color='gray',lw=1.5, linestyle=':',alpha=0.5)
    # ax2.text(zhaoqiao["Longitude"][5], zhaoqiao["Latitude"][5], 'Zhaoqiao', fontsize=8, ha='left', va='bottom', color='black')
    print("所有断层图已绘制完成。")
    # 触发概率图层
    prob_grid = griddata(
        (earthquakes[:, 0], earthquakes[:, 1]),
        prob_values,
        (data['lons'], data['lats']),
        method='linear'
    )
    
    im2 = ax2.pcolormesh(
        data['lons'], data['lats'], prob_grid,
        cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1,
        transform=ccrs.PlateCarree()
    )
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', shrink=0.6)
    cbar2.set_label('probability')
    # ax2.set_title('earthquake triggering probability distribution')
    
    # 绘制地震事件
    sc2 = ax2.scatter(
        earthquakes[:, 0], earthquakes[:, 1],
        c=prob_values, cmap='RdYlBu_r', vmin=0, vmax=1,
        s=5, edgecolors='k', linewidths=0.5, alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    # for name, (lat, lon) in cities.items():
    #     ax2.scatter(lon, lat, c='black', s=10, zorder=10)
    #     ax2.text(lon + 0.05, lat + 0.05, name, fontsize=8, ha='left', va='bottom', color='black')
    # 子图3: 地下水位变化
    # ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    # ax3.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    # ax3.add_feature(cfeature.COASTLINE)
    # ax3.add_feature(cfeature.BORDERS, linestyle=':')
    # ax3.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    # ax3.gridlines(draw_labels=True)
    
    # if groundwater is not None:
    #     im3 = ax3.pcolormesh(
    #         data['lons'], data['lats'], groundwater,
    #         cmap='coolwarm', shading='auto',
    #         transform=ccrs.PlateCarree()
    #     )
    #     cbar3 = plt.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.6)
    #     cbar3.set_label('water level (m)')
    #     ax3.set_title('water level variation')
    # else:
    #     ax3.text(0.5, 0.5, '无地下水位数据', 
    #             horizontalalignment='center', verticalalignment='center',
    #             transform=ax3.transAxes, fontsize=12)
    #     ax3.set_title('地下水位变化 (无数据)')
    
    # 子图4: 库仑应力率与触发概率关系
    ax4 = fig.add_subplot(224)
    valid_mask = np.isfinite(csr_values) & np.isfinite(prob_values)
    ax4.scatter(csr_values[valid_mask], prob_values[valid_mask], 
               c=earthquakes[valid_mask, 3], cmap='viridis', alpha=0.6)
    ax4.set_xlabel('CSR (kPa/yr)')
    ax4.set_ylabel('probality')
    # ax4.set_title('CSR& Triggering Probability')
    cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar4.set_label('magnitude')
    
    # 添加趋势线
    if np.sum(valid_mask) > 10:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            csr_values[valid_mask], prob_values[valid_mask])
        x = np.linspace(np.min(csr_values[valid_mask]), np.max(csr_values[valid_mask]), 100)
        ax4.plot(x, slope*x + intercept, 'r--', 
                label=f'R²={r_value**2:.2f}, p={p_value:.4f}')
        ax4.legend()
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.savefig('stress_trigger_results_update.svg', dpi=650)
    plt.show()
    


# 主工作流程
def main():
    # 加载数据
    print("=== 库仑应力验证工作流 ===")
    print(f"研究区域: {LON_MIN}-{LON_MAX}°E, {LAT_MIN}-{LAT_MAX}°N")
    print(f"网格分辨率: {GRID_RESOLUTION}°")
    print("1. 加载数据...")
    my_catalog = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/QJvalidation/hypoDD_final_allevents.reloc'
    gnss_file = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/Qiaojia/plotting/TableS1.xlsx'
    dilatatinion_file = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/Qiaojia/plotting/TableS2.xlsx'
    shear_strain_file = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/Qiaojia/plotting/TableS3.xlsx'
    rotation_file = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/Qiaojia/plotting/TableS4.xlsx'
    groundwater_file = '/Users/chouyuhin/_Workhome/Proj_Seismicgap/QJvalidation/蒙姑水位.txt'  # 替换为实际的地下水位数据文件
    earthquakes, gnss_data, dilatation, shear_strain, rotation,groundwater = load_data(my_catalog,gnss_file, dilatatinion_file, shear_strain_file, rotation_file,groundwater_file)
    
    print("2. 数据预处理...")   #
    rbf_ve,rbf_vn, grid_data = preprocess_data(gnss_data, dilatation, shear_strain, rotation,groundwater)
    
    print("3. 计算应变率张量...")
    exx, eyy, ezz, exy = calculate_strain_tensor(grid_data)
    
    print("4. 计算应力率张量...")
    sxx, syy, szz, sxy, sxz, syz = calculate_stress_rate_tensor(exx, eyy, ezz, exy,depth_grid=grid_data['depth'])
    
    print("5. 计算库仑应力率...")
    csr = calculate_coulomb_stress_rate(sxx, syy, szz, sxy, sxz, syz, grid_data)
    
    print("6. 计算小江断裂带上的库仑应力率且画图...")
    with open("./Xiaojiang_Fault_all.txt") as f:
        fault_lines = f.readlines()
    segment_dict = read_fault_segments(fault_lines)
    
    # # 计算所有断层段 ΔCFS
    results = compute_all_segments_cfs(
        segment_dict=segment_dict,
        grid_lon = grid_data['lons'],
        grid_lat = grid_data['lats'],
        sxx=sxx,
        syy=syy,
        szz=szz,
        sxy=sxy,
        sxz=sxz,
        syz=syz,
        strike_deg=167,
        dip_deg=85,
        rake_deg=-10,
        mu=0.4,
        half_width_km=2
    )
    # 3绘图
    plot_all_fault_segments(results, extent=[102.5, 104, 24, 28])
    
    print("7. 画出gnss模拟对比图...")
    # evaluate_velocity_model(gnss_data, exx, eyy, exy, grid_data)
    
    
    print("7. 画出gnss边界条件...")
    fig, ax = plt.subplots(figsize=(8, 8))
    bounds = (102, 104, 24, 28)

    # plot_velocity_field_with_boundaries(
    #     ax, rbf_ve, rbf_vn,
    #     bounds=bounds,
    #     obs_data=gnss_data,  # 若没有观测数据可设为 None
    #     scale=1/0.008
    # )
    # ax.set_xlim(bounds[0], bounds[1] + 0.1)  # 向右扩展
    # ax.set_ylim(bounds[2] - 0.1, bounds[3])  # 向下扩展
    # ax.set_title("GNSS boundaryconstraint", fontsize=12)
    # ax.set_aspect('equal')
    # plt.tight_layout()
    # plt.savefig('GNSS boundaryconstraint.svg', dpi=650)
    # plt.show()

    print("8. 验证地震目录...")
    ratio, p, csr_values,prob_values = validate_catalog(
        earthquakes, csr, grid_data['lons'], grid_data['lats'],grid_data['groundwater'])
    
    # plot_sensitivity_curves_2d(csr_values,grid_data['groundwater'])
    print("9. 生成可视化结果...")
    # plot_results(grid_data, csr, earthquakes, csr_values,prob_values,grid_data['groundwater'])
    
    # 返回关键结果
    return {
        'positive_ratio': ratio,
        'p_value': p,
        'mean_csr': np.nanmean(csr),
        'csr_values': csr_values,
        'prob_values': prob_values
        
    }

if __name__ == "__main__":
    results = main()