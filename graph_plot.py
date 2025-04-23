import matplotlib.pyplot as plt

# Sizes for x-axis
sizes = ["5e7", "8e7", "1e8", "3e8", "5e8"]

# Insert Throughput (in M ops/sec)
insert = {
    "BGHT": [1992.42, 1986.30, 1985.91, 1985.00, 1984.65],
    "SlabHash": [2019.63, 2015.61, 2014.35, 2011.20, None],
    "DyCuckoo": [553.21, 553.44, 553.42, 552.80, None],
    "WarpCore": [2064.30, 2014.25, 1982.34, 2147.95, 1980.88]
}

# Search Throughput (in M ops/sec)
search = {
    "BGHT": [3887.91, 3898.85, 3886.82, 3887.15, 3885.57],
    "SlabHash": [3896.41, 3897.48, 3897.13, 3896.37, None],
    "DyCuckoo": [2127.63, 2118.53, 2121.75, 2120.01, None],
    "WarpCore": [2114.77, 1999.52, 1956.18, 2341.21, 2341.71]
}

# Delete Throughput (in M ops/sec)
delete = {
    "BGHT": [1927.12, 1892.54, 1933.26, 1932.06, 1934.12],
    "SlabHash": [3872.68, 3871.01, 3871.10, 3870.10, None],
    "DyCuckoo": [2172.30, 2185.97, 2177.23, 2175.11, None],
    "WarpCore": [2106.36, 2104.25, 2102.94, 2298.01, 2297.59]
}

# Function to plot a category of throughput
def plot_throughput(data, title):
    plt.figure(figsize=(10, 6))
    for label, values in data.items():
        # Replace None with NaN to avoid plot errors
        plt.plot(sizes, [v if v is not None else float('nan') for v in values], marker='o', label=label)
    plt.title(f"{title} Throughput vs Size")
    plt.xlabel("Input Size")
    plt.ylabel("Throughput (M ops/sec)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generate plots
plot_throughput(insert, "Insert")
plot_throughput(search, "Search")
plot_throughput(delete, "Delete")
