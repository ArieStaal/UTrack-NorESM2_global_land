from netCDF4 import Dataset, date2index
import random
import numpy as np
import datetime as dt
import time as timer
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point


# Get index based on vertical position
def get_level_index(levels, level):
    if level > 925: return 0
    if level > 775: return 1
    if level > 600: return 2
    if level > 375: return 3
    if level > 175: return 4
    if level > 75: return 5
    if level > 30: return 6
    return 7

# Get index based on horizontal position
def get_pos_index(lat, lon):
    lat_idx = int(round((lat + 90) / (181 / 192)))
    lon_idx = int(round(lon / 1.25))
    if lon_idx > 287 or lon_idx < 0: lon_idx = 0
    return lat_idx, lon_idx

def create_parcels_mask(mask, lats, lons, parcel_start_time, m, delta_t, parcels_per_mm=1):
    m.E[m.E<0]=0
    total_evaporation=np.sum(mask*m.E) * delta_t.total_seconds()
    average_evaporation=total_evaporation/np.sum(mask>0)
    norm_mask=(m.E*mask)/total_evaporation
    norm_mask = np.asarray(norm_mask).astype('float64')
    if not np.sum(norm_mask) == 0:        
        norm_mask = norm_mask / np.sum(norm_mask)
        x=np.arange(192*288).reshape([192,288])
        num_parcels=int(average_evaporation*parcels_per_mm)
        xy=np.random.choice(np.matrix.flatten(x),num_parcels,p=np.matrix.flatten(norm_mask))
        indices = np.unravel_index(xy, x.shape)
        locs = np.c_[indices]
        latlons = []
        for i in range(locs.shape[0]):
            latlons.append((lats[locs[i][0]],lons[locs[i][1]]))
        if average_evaporation<=0:
            num_parcels=0
        parcel_list=[]
        for i in range(num_parcels):
            # Add a moisture parcel and distribute randomly within cell
            parcel_list.append(Parcel(latlons[i][0] + (random.random() * 0.9424 - 0.4712),
                                      latlons[i][1] + (random.random() * 1.25 - 0.625),
                                      parcel_start_time,
                                      m,
                                      parcel_moisture=total_evaporation / num_parcels))
    else:
        parcel_list=[]
    return parcel_list

degreelength_lat = []
degreelength_lon = []

ds = Dataset('pr_day_NorESM2-MM_ssp245_r1i1p1f1_gn_20150101-20201231.nc') # or any other NorESM2 data file
dis_lats = ds.variables['lat'][:].copy()
ds.close()

# Calculate degree lengths
for i in dis_lats:
    curlatrad = i * 2 * np.pi / 360
    degreelength_lon.append(
        ((111412.84 * np.cos(curlatrad)) - 93.5 * np.cos(3 * curlatrad) + 0.118 * np.cos(5 * curlatrad)))
    degreelength_lat.append((111132.92 + (-559.82 * np.cos(2 * curlatrad)) + 1.175 * np.cos(
        4 * curlatrad) - 0.0023 * np.cos(6 * curlatrad)))

# Parcel class
class Parcel:
    def __init__(self, startlat, startlon, starttime, m, parcel_moisture=1):
        self.lat = startlat
        self.lon = startlon
        if self.lat > 90: self.lat = 90
        if self.lat < -90: self.lat = -90
        if self.lon > 360: self.lon -= 360
        if self.lon < 0: self.lon += 360
        self.startlatidx, self.startlonidx = get_pos_index(self.lat, self.lon)
        self.time = starttime
        self.moisture_present = parcel_moisture
        self.original_moisture = parcel_moisture
        self.present_fraction = 1
        self.level = self.get_starting_level(m)
        self.tracked = True

    # Determine vertical position of the parcel based on humidity
    def get_starting_level(self, m):
        # Get index
        latidx, lonidx = get_pos_index(self.lat, self.lon)
        H_sum = 0
        # Calculate sum of humidity
        for i in range(len(m.levels)):
            if lonidx < 0 | lonidx > 287:
                print(lonidx)
            else:
                if m.H[i, latidx, lonidx] > 0:
                    H_sum += m.H[i, latidx, lonidx]
        # Generate random number based on sum of humidity
        frac = random.random() * H_sum
        count = 0
        # Go through pressure levels, and return a the point that
        # the cumulative humidity is higher than the random number
        for i in range(len(m.levels)):
            if m.H[i, latidx, lonidx] > 0:
                count += m.H[i, latidx, lonidx]
                if count > frac:
                    return m.levels[i]
        return 950

    # Progress moisture parcel
    def progress(self, delta_t, output, m):
        t1 = timer.perf_counter() * 100000
        # Get index and timestep in seconds
        delta_t_sec = delta_t.total_seconds()
        levidx = get_level_index(m.levels, self.level)
        latidx, lonidx = get_pos_index(self.lat, self.lon)

        # Get vertical tendency of air pressure
        w = m.w[levidx, latidx, lonidx]

        # Check level
        while np.abs(w) > 1000:
            self.level -= 50
            levidx = get_level_index(m.levels, self.level)
            w = m.w[levidx, latidx, lonidx]

        # Get wind speeds
        u = m.u[levidx, latidx, lonidx]
        v = m.v[levidx, latidx, lonidx]

        # Get degree lengths
        dlat = degreelength_lat[latidx]
        dlon = degreelength_lon[latidx]

        # Move parcel
        self.lon += (delta_t_sec * u / dlon)
        self.lat += (delta_t_sec * v / dlat)
        if self.lat > 90: self.lat = 89
        if self.lat < -90: self.lat = -89
        if self.lon > 360: self.lon -= 360
        if self.lon < 0: self.lon += 360
        self.level += (delta_t_sec * w / 100)
        if self.level < 50: self.level = 50
        if self.level > 1000: self.level = 1000

        # Get index
        latidx, lonidx = get_pos_index(self.lat, self.lon)

        # Randomly distribute moisture vertically, with an expectation of doing this once per day
        frac = random.random() * 24

        if frac < (delta_t_sec / 3600):
            self.level = self.get_starting_level(m)

        # Make moisture budget
        P = m.P[latidx, lonidx] * delta_t_sec
        PW = m.TCWV[latidx, lonidx]
        fraction_allocated = P / PW
        currently_allocated = fraction_allocated * self.moisture_present
        self.moisture_present -= currently_allocated
        self.present_fraction = self.moisture_present / self.original_moisture

        # Allocate precipitation
        outlatidx, outlonidx = get_pos_index(self.lat, self.lon)

        try:
            output[outlatidx, outlonidx] += currently_allocated
        except:
            print(outlatidx, outlonidx)

        return output


# Forcing data class
class Meteo:
    def __init__(self):

        self.time = 0
        self.levels = 0
        self.lats = 0
        self.lons = 0
        self.u = 0
        self.v = 0
        self.E = 0
        self.P = 0
        self.H = 0
        self.TCWV = 0
        self.w = 0

    def read_from_file(self, year, month, day, scenario, forcing_directory):
        if 2015 <= year <= 2020:
            yearstr = '20150101-20201231'
        elif 2021 <= year <= 2030:
            yearstr = '20210101-20301231'
        elif 2031 <= year <= 2040:
            yearstr = '20310101-20401231'
        elif 2041 <= year <= 2050:
            yearstr = '20410101-20501231'
        elif 2051 <= year <= 2060:
            yearstr = '20510101-20601231'
        elif 2061 <= year <= 2070:
            yearstr = '20610101-20701231'
        elif 2071 <= year <= 2080:
            yearstr = '20710101-20801231'
        elif 2081 <= year <= 2090:
            yearstr = '20810101-20901231'
        elif 2091 <= year <= 2100:
            yearstr = '20910101-21001231'

        e = Dataset(
            forcing_directory + 'evspsbl_Eday_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        p = Dataset(
            forcing_directory + 'pr_day_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        h = Dataset(
            forcing_directory + 'hus_day_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        u = Dataset(
            forcing_directory + 'ua_day_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        v = Dataset(
            forcing_directory + 'va_day_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        tcwv = Dataset(
            forcing_directory + 'prw_Eday_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')
        wap = Dataset(
            forcing_directory + 'wap_day_NorESM2-MM_ssp'
            + str(scenario) + '_r1i1p1f1_gn_' + yearstr + '.nc')

        time = u.variables['time']
        time_idx = date2index(dt.datetime(year, month, day), time, calendar='noleap', select='nearest')

        self.time = time
        self.levels = u.variables['plev'][:].copy() / 100
        self.lats = u.variables['lat'][:].copy()
        self.lons = u.variables['lon'][:].copy()
        self.u = u.variables['ua'][time_idx].copy()
        self.v = v.variables['va'][time_idx].copy()
        self.E = e.variables['evspsbl'][time_idx].copy()
        self.P = p.variables['pr'][time_idx].copy()
        self.H = h.variables['hus'][time_idx].copy()
        self.TCWV = tcwv.variables['prw'][time_idx].copy()
        self.w = wap.variables['wap'][time_idx].copy()

        e.close()
        p.close()
        h.close()
        u.close()
        v.close()
        tcwv.close()
        wap.close()

