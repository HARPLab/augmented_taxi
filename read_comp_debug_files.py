import pstats
from pstats import SortKey
import pandas as pd
import csv



p = pstats.Stats('sim_profiler.txt')
print(type(p))
# p.strip_dirs().sort_stats().print_stats()

df = pd.DataFrame(p.stats).T

# Rename columns for clarity
df.columns = ['ncalls', 'tottime', 'cumtime', 'filename', 'function']

# Convert columns to numeric where applicable
numeric_cols = ['ncalls', 'tottime', 'cumtime']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

df = df.drop('function', axis=1)

# print(df.columns)

# df = pd.DataFrame(p.strip_dirs().sort_stats('time').print_stats())
df.to_csv('sim_profiler_df.csv')

# profiling_data = []
# for func_info in p.stats:
#     print('func_info: ', func_info)
#     file_name, line_number, function_name = func_info[:3]
#     cumulative_time, tottime, pcalls, ncalls, time = p.stats[func_info]

#     print('cumulative_time: ', cumulative_time)
#     print('tottime: ', tottime)
#     print('pcalls: ', pcalls)
#     print('ncalls: ', ncalls)
#     print('time: ', time)

#     # Add more fields as needed based on your requirements
#     row_data = {
#         'File': file_name,
#         'Line': line_number,
#         'Function': function_name,
#         'Cumulative Time': cumulative_time,
#         'tottime': tottime,
#         'pcalls': pcalls,
#         'ncalls': ncalls, 
#         'time': time
#     }

#     profiling_data.append(row_data)



# # Write the data to a CSV file
# with open('sim_profiler.csv', 'w', newline='') as csv_file:
#     fieldnames = ['File', 'Line', 'Function', 'Cumulative Time', 'tottime', 'pcalls', 'ncalls', 'time']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     # Write the header
#     writer.writeheader()

#     # Write the data
#     writer.writerows(profiling_data)





#########################################

# import cProfile
# import pstats, math
# import io
# import pandas as pd
 
# pr = cProfile.Profile()
# pr.enable()
# print(math.sin(1024))
# pr.disable()
 
# result = io.StringIO()
# pstats.Stats(pr,stream=result).print_stats()
# result=result.getvalue()
# # chop the string into a csv-like buffer
# result='ncalls'+result.split('ncalls')[-1]
# result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
# # save it to disk
 
# with open('test.csv', 'w+') as f:
#     #f=open(result.rsplit('.')[0]+'.csv','w')
#     f.write(result)
#     f.close()