import pandas as pd

def separate_and_count_programs(df, program_list):
    program_dfs = []
    for program in program_list:
        program_df = df[df.program == program]
        print(program_df['user_id'].nunique(), "users in", program)
        print(len(program_df), "trips in", program)
        program_dfs.append(program_df)
    
    return program_dfs

def filter_before_ebike(program_df):
    #timestamp conversion
    program_df['start_ts']= pd.to_datetime(program_df['start_ts'], utc=True, unit='s')

    #grouping, counting unique users
    trip_sep=program_df.groupby(['user_id','Mode_confirm']).apply(lambda x:x[x.start_ts==min(x.start_ts)])
    print(trip_sep['user_id'].nunique(), "users before filtering")

    #consider only trips with E-bike (to get first e-bike trip)
    ebike_first=trip_sep[trip_sep['Mode_confirm']=='E-bike']

    #get all the trips by ysers who ever had an e-bike trip
    ebike_user_list= ebike_first['user_id'].tolist()
    incl_ebike = program_df[program_df['user_id'].isin(ebike_user_list)]
    print(incl_ebike['user_id'].nunique(), "users who traveled by ebike")

    #filter to the earliest ebike trip
    for unique_id in ebike_first['user_id']:
        for date in ebike_first['start_ts']:
            filtered_ebike_first=incl_ebike[(incl_ebike['start_ts'] >= date)]

    ebikefirst=filtered_ebike_first['user_id'].unique()

    return filtered_ebike_first