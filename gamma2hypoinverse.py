#%%
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# %%
picks = pd.read_csv('D:/MyRepository/Qiaojia/AssResults_modifydepth/gamma_picks.csv', sep="\t")
events = pd.read_csv('D:/MyRepository/Qiaojia/AssResults_modifydepth/gamma_catalog.csv', sep="\t")

# %%
events["match_id"] = events["event_index"]
picks["match_id"] = picks["event_index"]

# %%
out_file = open("hypoInput.arc", "w",encoding='ascii')

picks_by_event = picks.groupby("match_id").groups

for i in tqdm(range(len(events))):

    event = events.iloc[i]
    # print(event)
    event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y%m%d%H%M%S%f")[:-4]       #time: 2023-02-06T20:26:32.232	2023020620263223	
    # print(event_time)
    lat_degree = int(event["latitude"])
    # print(lat_degree)
    lat_minute = (event["latitude"] - lat_degree) * 60 * 100
    south = "S" if lat_degree <= 0 else "N"
    # print(f'south={south}')
    lng_degree = int(event["longitude"])
    lng_minute = (event["longitude"] - lng_degree) * 60 * 100
    east = "E" if lng_degree >= 0 else " "
    depth = event["depth(m)"] / 1e3 * 100
    event_line = f"{event_time}{abs(lat_degree):2d}{south}{abs(lat_minute):4.0f}{abs(lng_degree):3d}{east}{abs(lng_minute):4.0f}{depth:5.0f}"
    out_file.write(event_line + "\n")

    picks_idx = picks_by_event[event["match_id"]]
    for j in picks_idx:
        pick = picks.iloc[j]
        network_code, station_code,comp_code, channel_code= pick['id'][:2], pick['id'][2:],'','99'
        phase_type = pick['type']
        phase_weight = min(max(int((1 - pick['prob']) / (1 - 0.3) * 4) - 1, 0), 3)
        pick_time = datetime.strptime(pick["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
        phase_time_minute = pick_time.strftime("%Y%m%d%H%M")
        phase_time_second = pick_time.strftime("%S%f")[:-4]
        tmp_line = f"{station_code:<5}{network_code:<2} {comp_code:<1}{channel_code:<3}"
        if phase_type.upper() == 'P':
            pick_line = f"{tmp_line:<13} P {phase_weight:<1d}{phase_time_minute} {phase_time_second}"
        elif phase_type.upper() == 'S':
            pick_line = f"{tmp_line:<13}   4{phase_time_minute} {'':<12}{phase_time_second} S {phase_weight:<1d}"
        else:
            raise (f"Phase type error {phase_type}")
        out_file.write(pick_line + "\n")

    out_file.write("\n")
    # if i > 10:
    #     break      # 实际运行时删除

out_file.close()
