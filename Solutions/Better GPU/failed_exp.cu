#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cassert>
#include <chrono>
#include "bght/bcht.hpp"

#define DEVICE_ID 0

using KeyT = uint32_t;
using ValueT = uint32_t;
using HashTableT = bght::bcht<KeyT, ValueT>;

// Function to read binary data from file (for insert, search, delete operations)
std::vector<KeyT> read_binary_file(const std::string& path, size_t max_elements) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = std::min(size / sizeof(KeyT), max_elements);
    std::vector<KeyT> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(KeyT));
    return data;
}

    
int main(int argc, char* argv[]) {
    cudaSetDevice(DEVICE_ID);
    
    std::unordered_map<std::string, std::string> args;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        size_t pos = arg.find("="); // Find '=' separator

        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);   // Extract key (e.g., "--size")
            std::string value = arg.substr(pos + 1); // Extract value (e.g., "1000")
            args[key] = value; // Store in map
        }
    }

    bool verbose = args.count("--verbose") ? std::atoi(args["--verbose"].c_str()) : 1;
    int size = args.count("--size") ? std::atoi(args["--size"].c_str()) : 1e9;
    double ins = args.count("--ins") ? std::atof(args["--ins"].c_str()) : 0.0;
    double lkp = args.count("--lkp") ? std::atof(args["--lkp"].c_str()) : 0.0;
    double rem = args.count("--rem") ? std::atof(args["--rem"].c_str()) : 0.0;

    size_t MAX_INSERT = size * ins;
    size_t MAX_SEARCH = size * lkp;
    size_t MAX_DELETE = size * rem;

    
    // Paths to trace files
    std::string insert_path = "/data/vipinpat/trace-files/insert_trace-400e7-100-add-no-dup.bin";
    std::string search_path = "/data/vipinpat/trace-files/search_trace-400e7-30-find-no-dup-no-absent.bin";
    std::string delete_path = "/data/vipinpat/trace-files/delete_trace-400e7-10-rem-no-dup-no-absent.bin";
    
    // Load the trace files
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<KeyT> insert_keys = read_binary_file(insert_path, MAX_INSERT);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    if (verbose) std::cout << "Loaded " << insert_keys.size() << " elements in " 
              << duration.count() << " ms" << std::endl;
    
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<KeyT> search_keys = read_binary_file(search_path, MAX_SEARCH);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    if (verbose) std::cout << "Loaded " << search_keys.size() << " elements in " 
              << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    std::vector<KeyT> delete_keys = read_binary_file(delete_path, MAX_DELETE);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    if (verbose) std::cout << "Loaded " << delete_keys.size() << " elements in " 
              << duration.count() << " ms" << std::endl;
    
    // Prepare device vectors for insert and search operations
    thrust::device_vector<bght::pair<KeyT, ValueT>> d_insert_pairs(insert_keys.size());
    for (size_t i = 0; i < insert_keys.size(); ++i) {
        d_insert_pairs[i] = {insert_keys[i], static_cast<ValueT>(i)};
    }
    
    thrust::device_vector<KeyT> d_search_keys(search_keys);
    thrust::device_vector<KeyT> d_delete_keys(delete_keys);
    thrust::device_vector<ValueT> d_results(search_keys.size());
    
    // Define sentinel values (invalid key-value)
    KeyT sentinel_key = std::numeric_limits<KeyT>::max();
    ValueT sentinel_value = std::numeric_limits<ValueT>::max();
    
    
    // Create the hash table
    printf("Hellp");
    
    start_time = std::chrono::high_resolution_clock::now();
    HashTableT hash_table(2 * insert_keys.size(), sentinel_key, sentinel_value);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    
    printf("Creation Time: %.3f ms\n", duration.count());
    
    if (verbose) std::cout << "Initializing GPU Hash Table with " << insert_keys.size() 
              << " keys and " << 2 * insert_keys.size() << " buckets" << std::endl;
    
    // Insert data into hash table
    if (verbose) std::cout << "Starting insertion of " << insert_keys.size() << " elements..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    bool success = hash_table.insert(d_insert_pairs.data().get(), 
                                     d_insert_pairs.data().get() + d_insert_pairs.size());
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    assert(success && "Insertion failed!");
    double insertion_rate = static_cast<double>(insert_keys.size()) / duration.count() * 1000.0; // M elements/s
    // std::cout << "Insertion Time: " << duration.count() << " ms (" << insertion_rate << " M elements/s)" << std::endl;
    printf("Insertion Time: %.3f ms (%.3f M elements/s)\n", duration.count(), insertion_rate / 1e6);

    // Perform search queries
    if (verbose) std::cout << "Starting search of " << search_keys.size() << " elements..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    hash_table.find(d_search_keys.data().get(), 
                    d_search_keys.data().get() + d_search_keys.size(), 
                    d_results.begin());
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    double search_rate = static_cast<double>(search_keys.size()) / duration.count() * 1000.0; // M elements/s
    thrust::host_vector<ValueT> h_results = d_results;
    assert(h_results.size() == search_keys.size());
    // std::cout << "Search Time: " << duration.count() << " ms (" 
            //   << search_rate << " M elements/s)" << std::endl;
    printf("Search Time: %.3f ms (%.3f M elements/s)\n", duration.count(), insertion_rate / 1e6);


    // Prepare delete operations (set to sentinel value)
    if (verbose) std::cout << "Starting deletion of " << delete_keys.size() << " elements..." << std::endl;
    thrust::device_vector<bght::pair<KeyT, ValueT>> d_delete_pairs(delete_keys.size());
    for (size_t i = 0; i < delete_keys.size(); ++i) {
        d_delete_pairs[i] = {delete_keys[i], sentinel_value};
    }
    
    // Insert delete operations to set to sentinel value
    start_time = std::chrono::high_resolution_clock::now();
    success = hash_table.insert(d_delete_pairs.data().get(), 
                                d_delete_pairs.data().get() + d_delete_pairs.size());
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    double delete_rate = static_cast<double>(delete_keys.size()) / duration.count() * 1000.0; // M elements/s
    assert(success && "Deletion failed!");
    // std::cout << "Deletion Time: " << duration.count() << " ms (" 
            //   << delete_rate << " M elements/s)" << std::endl;
    printf("Deletion Time: %.3f ms (%.3f M elements/s)\n", duration.count(), insertion_rate / 1e6);

    // Calculate load factor
    double load_factor = static_cast<double>(insert_keys.size()) / (2 * insert_keys.size());
    std::cout << "Load factor: " << load_factor << std::endl;
    
    // Final success message
    if (verbose) std::cout << "Program completed successfully!" << std::endl;
    return 0;
}