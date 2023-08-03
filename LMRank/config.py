import psutil

# Find all logical cores (threads) of the system.
all_threads = psutil.cpu_count(logical = True)

# Find all logical cores (threads) available to the program.
available_threads = len(psutil.Process().cpu_affinity())

# The utilized threads are set to a fraction (1/3) of total threads
# or all available ones, if the latter are fewer.
thread_fraction = all_threads // 3

if thread_fraction < 1:
    num_threads = f'{min(all_threads, available_threads)}'
elif available_threads < thread_fraction:
    num_threads = f'{available_threads}'
else:
    num_threads = f'{thread_fraction}'
