# Imports
import netCDF4 as nc
import datetime
from UTrack_NorESM2_global_land_functions import *
import time

t1 = time.perf_counter()

# Directory where NorESM2 data are stored
forcing_directory = ''

# SIMULATION SETTINGS
yr = 2015 # Year for which the simulation should be run
mo = 1 # Month for which the simulation should be run
scenario = 245 # SSP of the simulation (126, 245, 370 or 585)
parcels_mm = 1000  # Number of parcels to be released per mm evaporation

# Load global land mask
fb=Dataset('land_mask_NorESM2.nc')
land_mask=fb.variables['land'][:].copy()

# Set temporal simulation settings based on the simulation year and month
days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
start_date = datetime.datetime(yr, mo, 1)
if mo == 11:
    end_date = datetime.datetime(yr+1, 1, 1)
elif mo == 12:
    end_date = datetime.datetime(yr+1, 1, 31, 23)
else:
    end_date = datetime.datetime(yr, mo+2, 1)
release_end_date = datetime.datetime(yr, mo, days_in_months[mo-1], 23)
dt = datetime.timedelta(hours=4)  # Set timestep length

# INITIALISATION
# Set the model timer
current_date = start_date

# Create empty meteo objects for holding forcing data
m = Meteo()
m1 = Meteo()
m2 = Meteo()

# Load forcing data for the current day and the day after
nextday = current_date + datetime.timedelta(days=1)
m1.read_from_file(current_date.year, current_date.month, current_date.day, scenario, forcing_directory)
m2.read_from_file(current_date.year, current_date.month, current_date.day + 1, scenario, forcing_directory)

# Define variables for runtime estimation
runtimes = []
time_estimate = 0
num_days = (end_date - start_date).days

# Create output file if one does not exist already
outputfn = 'SSP'+str(scenario)+'/utrack-noresm2_forw_global_land_ssp'+str(scenario)+'_'+str(yr)+'-'+str(mo).zfill(2)+'.nc'
output = nc.Dataset(outputfn,mode='w',format='NETCDF4_CLASSIC')

output.title = 'Evaporation footprints'

lat_dim = output.createDimension('lat', len(m1.lats))
lon_dim = output.createDimension('lon', len(m1.lons))
src_lat_dim = output.createDimension('srclat', len(m1.lats))
src_lon_dim = output.createDimension('srclon', len(m1.lons))
frac_dim = output.createDimension('frac',1)

lat = output.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = output.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
value = output.createVariable('footprint', np.float64, ('lat', 'lon'), zlib=True)
value.units = 'mm'
value.long_name = 'Amount of precipitation coming from source'
fraction_allocated = output.createVariable('fraction_allocated', np.float32 , 'frac')
fraction_allocated.units = 'fraction'
fraction_allocated.long_name = 'Fraction of total released moisture allocated to precipitation'

lat[:] = m1.lats
lon[:] = m1.lons
rlocs = []

output.close()

t3 = time.perf_counter()

# SIMULATION
print(f'Year: {yr}, Month: {mo}, SSP{scenario}.')#', Cells to go: {str(to_go)}.')

# Create output array with the dimensions of the forcing data
output_array = np.zeros([len(m1.lats), len(m1.lons)])
# Create empty list of moisture parcels
parcels = []
# Define tracking variables
parcelsin = 0
killed = 0
expired = 0
total_released = 0
days_elapsed = 0
# Set minimum amount of moisture present in a parcel
kill_threshold = 0.01
# Set model timer
current_date = start_date
# Load forcing data
m = Meteo()

print(
    "Date   Time    Parcels in system   Parcels added   Parcels destroyed  Simulation time of time step (s)  Fraction of moisture allocated")
# Repeat for each time step
while current_date <= end_date:
    # Skip the interation if the model timer is equal to a leap day
    if current_date.month == 2 and current_date.day == 29:
        current_date += dt
        continue

    # Load forcing data
    if current_date.hour == 0:
        m.read_from_file(current_date.year, current_date.month, current_date.day, scenario, forcing_directory)

    # Release moisture parcels
    if current_date<=release_end_date:
        plist = create_parcels_mask(land_mask, m1.lats, m1.lons, current_date, m, dt, parcels_mm)
        parcels.extend(plist)
        total_released += np.sum([p.original_moisture for p in plist])

    # Progress moisture parcels
    for p in parcels:
        output_array = p.progress(dt, output_array, m)

    # Destroy moisture parcels
    prevlength = len(parcels)
    parcels = [p for p in parcels if p.present_fraction >= kill_threshold] # destroy parcel if 99% allocated to precipitation
    intlength = len(parcels)
    parcels = [p for p in parcels if (current_date - p.time).total_seconds() < 2592000] # destroy parcel if tracked for 30 days
    killed_dt = prevlength - intlength
    killed += killed_dt
    expired_dt = intlength - len(parcels)
    expired += expired_dt

    # Print internal state of the model and estimated remaining runtime
    if current_date.hour == 0:
        print(current_date,
              len(parcels),
              len(parcels) - parcelsin,
              killed,
              expired,
              np.sum(output_array) / total_released)
        parcelsin = len(parcels)
        killed = 0
        expired = 0

    # Progress time
    current_date += dt

# Write data to output file
try:
    output.close(),
except:
    pass

output = nc.Dataset(outputfn, 'r+')
output_array = output_array/(np.sum(output_array)/total_released) # correct for incomplete allocation
output['footprint'][:, :] = output_array
output['fraction_allocated'][:] = np.sum(output_array)/total_released

output.close()

t2 = time.perf_counter()
print(f'Finished in {t2 - t1} seconds')
