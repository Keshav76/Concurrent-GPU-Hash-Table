#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cassert>
#include <chrono>
#include <unordered_map>
#include <cuda_runtime.h>
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

    assert(args.count("--insert_path") && "Insert path is required");
    assert(args.count("--search_path") && "Search path is required");
    assert(args.count("--delete_path") && "Delete path is required");
    assert(args.count("--size") && "Size is required");
    assert(args.count("--ins") && "Insert ratio is required");
    assert(args.count("--lkp") && "Lookup ratio is required");
    assert(args.count("--rem") && "Remove ratio is required");

    bool verbose = args.count("--verbose") ? std::atoi(args["--verbose"].c_str()) : 1;
    int size = args.count("--size") ? std::atoi(args["--size"].c_str()) : 1e9;
    double ins = args.count("--ins") ? std::atof(args["--ins"].c_str()) : 0.0;
    double lkp = args.count("--lkp") ? std::atof(args["--lkp"].c_str()) : 0.0;
    double rem = args.count("--rem") ? std::atof(args["--rem"].c_str()) : 0.0;
    std::string insert_path = args.count("--insert_path") ? args["--insert_path"] : "insert_trace-500e7-50-add-10-dup.bin";
    std::string search_path = args.count("--search_path") ? args["--search_path"] : "search_trace-500e7-40-find-20-dup-no-absent.bin";
    std::string delete_path = args.count("--delete_path") ? args["--delete_path"] : "delete_trace-500e7-10-rem-40-dup-no-absent.bin";

    size_t MAX_INSERT = size * ins;
    size_t MAX_SEARCH = size * lkp;
    size_t MAX_DELETE = size * rem;

    // Paths to trace files
    insert_path = "/data/vipinpat/trace-files/" + insert_path;
    search_path = "/data/vipinpat/trace-files/" + search_path;
    delete_path = "/data/vipinpat/trace-files/" + delete_path;
    
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

    // Allocate device memory for insert, search, and delete keys
    KeyT* d_insert_keys;
    KeyT* d_search_keys;
    KeyT* d_delete_keys;
    cudaMalloc(&d_insert_keys, insert_keys.size() * sizeof(KeyT));
    cudaMalloc(&d_search_keys, search_keys.size() * sizeof(KeyT));
    cudaMalloc(&d_delete_keys, delete_keys.size() * sizeof(KeyT));

    // Copy data from host to device
    cudaMemcpy(d_insert_keys, insert_keys.data(), insert_keys.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_search_keys, search_keys.data(), search_keys.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_delete_keys, delete_keys.data(), delete_keys.size() * sizeof(KeyT), cudaMemcpyHostToDevice);

    // Prepare insert pairs on host
    std::vector<bght::pair<KeyT, ValueT>> h_insert_pairs(insert_keys.size());
    for (size_t i = 0; i < insert_keys.size(); ++i) {
        h_insert_pairs[i] = {insert_keys[i], static_cast<ValueT>(i)};
    }

    // Allocate and copy insert pairs to device
    bght::pair<KeyT, ValueT>* d_insert_pairs;
    cudaMalloc(&d_insert_pairs, insert_keys.size() * sizeof(bght::pair<KeyT, ValueT>));
    cudaMemcpy(d_insert_pairs, h_insert_pairs.data(), insert_keys.size() * sizeof(bght::pair<KeyT, ValueT>), cudaMemcpyHostToDevice);

    // Define sentinel values (invalid key-value)
    KeyT sentinel_key = std::numeric_limits<KeyT>::max();
    ValueT sentinel_value = std::numeric_limits<ValueT>::max();
    
    // Create the hash table
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
    bool success = hash_table.insert(d_insert_pairs, d_insert_pairs + insert_keys.size());
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    assert(success && "Insertion failed!");
    double insertion_rate = static_cast<double>(insert_keys.size()) / duration.count() * 1000.0; // M elements/s
    printf("Insertion Time: %.3f ms (%.3f M elements/s)\n", duration.count(), insertion_rate / 1e6);

    // Allocate memory for search results on the device
    ValueT* d_results;
    cudaMalloc(&d_results, search_keys.size() * sizeof(ValueT));

    // Perform search queries
    if (verbose) std::cout << "Starting search of " << search_keys.size() << " elements..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    hash_table.find(d_search_keys, d_search_keys + search_keys.size(), d_results);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    double search_rate = static_cast<double>(search_keys.size()) / duration.count() * 1000.0; // M elements/s

    // Copy the results back to host for validation
    std::vector<ValueT> h_results(search_keys.size());
    cudaMemcpy(h_results.data(), d_results, search_keys.size() * sizeof(ValueT), cudaMemcpyDeviceToHost);

    assert(h_results.size() == search_keys.size());
    printf("Search Time: %.3f ms (%.3f M elements/s)\n", duration.count(), search_rate / 1e6);

    // Prepare delete pairs on host
    std::vector<bght::pair<KeyT, ValueT>> h_delete_pairs(delete_keys.size());
    for (size_t i = 0; i < delete_keys.size(); ++i) {
        h_delete_pairs[i] = {delete_keys[i], sentinel_value};
    }

    // Allocate and copy delete pairs to device
    bght::pair<KeyT, ValueT>* d_delete_pairs;
    cudaMalloc(&d_delete_pairs, delete_keys.size() * sizeof(bght::pair<KeyT, ValueT>));
    cudaMemcpy(d_delete_pairs, h_delete_pairs.data(), delete_keys.size() * sizeof(bght::pair<KeyT, ValueT>), cudaMemcpyHostToDevice);

    // Insert delete operations to set to sentinel value
    if (verbose) std::cout << "Starting deletion of " << delete_keys.size() << " elements..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    success = hash_table.insert(d_delete_pairs, d_delete_pairs + delete_keys.size());
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    double delete_rate = static_cast<double>(delete_keys.size()) / duration.count() * 1000.0; // M elements/s
    printf("Deletion Time: %.3f ms (%.3f M elements/s)\n", duration.count(), delete_rate / 1e6);

    // Cleanup
    cudaFree(d_insert_keys);
    cudaFree(d_search_keys);
    cudaFree(d_delete_keys);
    cudaFree(d_insert_pairs);
    cudaFree(d_delete_pairs);
    cudaFree(d_results);

    // Calculate load factor
    double load_factor = static_cast<double>(insert_keys.size()) / (2 * insert_keys.size());
    std::cout << "Load factor: " << load_factor << std::endl;
    
    // Final success message
    if (verbose) std::cout << "Program completed successfully!" << std::endl;
    return 0;
}
