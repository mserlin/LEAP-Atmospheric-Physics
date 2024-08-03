from data_utils import *

#Define input and output column names
#Some inputs and outputs are vertically resolved, corresponding to sequence data with a length of 60
#Others are simply tabular 
v2_inputs = ['state_t',
             'state_q0001',
             'state_q0002',
             'state_q0003',
             'state_u',
             'state_v',
             'state_ps',
             'pbuf_SOLIN',
             'pbuf_LHFLX',
             'pbuf_SHFLX',
             'pbuf_TAUX',
             'pbuf_TAUY',
             'pbuf_COSZRS',
             'cam_in_ALDIF',
             'cam_in_ALDIR',
             'cam_in_ASDIF',
             'cam_in_ASDIR',
             'cam_in_LWUP',
             'cam_in_ICEFRAC',
             'cam_in_LANDFRAC',
             'cam_in_OCNFRAC',
             'cam_in_SNOWHICE',
             'cam_in_SNOWHLAND',
             'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
             'pbuf_CH4',
             'pbuf_N2O']

v2_outputs = ['ptend_t',
              'ptend_q0001',
              'ptend_q0002',
              'ptend_q0003',
              'ptend_u',
              'ptend_v',
              'cam_out_NETSW',
              'cam_out_FLWDS',
              'cam_out_PRECSC',
              'cam_out_PRECC',
              'cam_out_SOLS',
              'cam_out_SOLL',
              'cam_out_SOLSD',
              'cam_out_SOLLD']

vertically_resolved = ['state_t', 
                       'state_q0001', 
                       'state_q0002', 
                       'state_q0003', 
                       'state_u', 
                       'state_v', 
                       'pbuf_ozone', 
                       'pbuf_CH4', 
                       'pbuf_N2O', 
                       'ptend_t', 
                       'ptend_q0001', 
                       'ptend_q0002', 
                       'ptend_q0003', 
                       'ptend_u', 
                       'ptend_v']

ablated_vars = ['ptend_q0001',
                'ptend_q0002',
                'ptend_q0003',
                'ptend_u',
                'ptend_v']

v2_vars = v2_inputs + v2_outputs

train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + '_' + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + '_' + str(i))
    else:
        train_col_names.append(var)

input_col_names = []
for var in v2_inputs:
    if var in vertically_resolved:
        for i in range(60):
            input_col_names.append(var + '_' + str(i))
    else:
        input_col_names.append(var)

output_col_names = []
for var in v2_outputs:
    if var in vertically_resolved:
        for i in range(60):
            output_col_names.append(var + '_' + str(i))
    else:
        output_col_names.append(var)

#Check that the length of the training columns, the input and the output data is all appropriate
assert(len(train_col_names) == 17 + 60*9 + 60*6 + 8)
assert(len(input_col_names) == 17 + 60*9)
assert(len(output_col_names) == 60*6 + 8)
assert(len(set(output_col_names).intersection(set(ablated_col_names))) == len(ablated_col_names))

#Path to the locally downloaded copy of the high-res repository 
DATA_PATH = 'C:/kaggle/CD L1/'

#Load the data grid info
grid_path = DATA_PATH+'ClimSim_high-res_grid-info.nc'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(DATA_PATH + 'input_mean.nc')
input_max = xr.open_dataset(DATA_PATH + 'input_max.nc')
input_min = xr.open_dataset(DATA_PATH + 'input_min.nc')
output_scale = xr.open_dataset(DATA_PATH + 'output_scale.nc')

# initialize data_utils object
data = data_utils(grid_info = grid_info, 
                  input_mean = input_mean, 
                  input_max = input_max, 
                  input_min = input_min, 
                  output_scale = output_scale)

data.set_to_v2_vars()

# do not normalize
data.normalize = False

#i is the 'year' index. The high-res data has up to 8 years of data. 
i = 5 # 6,7,8
#j is the month index, which gets loops through
for j in tqdm(range(1,13)):
    #Set the file names appropriately for the given year
    if j < 10:
        data.set_regexps(data_split = 'train', 
                         regexps = ['E3SM-MMF.mli.000'+str(i)+'-0'+str(j)+'-*-*.nc']) # first month of year 8
    else:
        data.set_regexps(data_split = 'train', 
                         regexps = ['E3SM-MMF.mli.000'+str(i)+'-'+str(j)+'-*-*.nc']) # first month of year 8
    
    # set temporal subsampling
    data.set_stride_sample(data_split = 'train', stride_sample = 1)
    # data.set_stride_sample()
    # create list of files to extract data from
    data.set_filelist(data_split = 'train')
    
    # save numpy files of training data
    data_loader = data.load_ncdata_with_generator(data_split = 'train')
    npy_iterator = list(data_loader.as_numpy_iterator())
    npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
    npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
    train_npy = np.concatenate([npy_input, npy_output], axis = 1)
    train_index = ["train_" + str(x) for x in range(train_npy.shape[0])]
    
    train = pd.DataFrame(train_npy, index = train_index, columns = train_col_names)
    train.index.name = 'sample_id'
    print('dropping cam_in_SNOWHICE because of strange values')
    train.drop('cam_in_SNOWHICE', axis=1, inplace=True)

    #save the file as a parquet
    train.to_parquet(DATA_PATH+'c_data_'+str(i)+'_'+str(j)+'.parquet')
