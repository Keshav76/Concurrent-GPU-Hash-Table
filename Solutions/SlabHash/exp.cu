#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include "./gpu_hash_table.cuh"
#include <cuda_runtime.h>

#define DEVICE_ID 0

using KeyT = uint32_t;
using ValueT = uint32_t;

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
    assert(file.gcount() == num_elements * sizeof(KeyT));
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

    insert_path = "/data/vipinpat/trace-files/" + insert_path;
    search_path = "/data/vipinpat/trace-files/" + search_path;
    delete_path = "/data/vipinpat/trace-files/" + delete_path;
    
    std::vector<KeyT> insert_keys = read_binary_file(insert_path, MAX_INSERT);
    std::vector<KeyT> search_keys = read_binary_file(search_path, MAX_SEARCH);
    std::vector<KeyT> remove_keys = read_binary_file(delete_path, MAX_DELETE);
    
    uint32_t num_keys = insert_keys.size();
    uint32_t num_buckets = num_keys / 0.5;
    
    if (verbose) printf("Initializing GPU Hash Table with %d keys and %d buckets\n", num_keys, num_buckets);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    gpu_hash_table<KeyT, ValueT, SlabHashTypeT::ConcurrentMap> hash_table(num_keys, num_buckets, DEVICE_ID, 1);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    printf("Creation Time: %.3f ms\n", duration.count());
    
    if (verbose) printf("Starting insertion of %zu elements...\n", insert_keys.size());
    float insert_time = hash_table.hash_build(insert_keys.data(), insert_keys.data(), num_keys);
    printf("Insertion Time: %.3f ms (%.3f M elements/s)\n", insert_time, double(num_keys) / insert_time / 1000.0);
    assert(num_keys == insert_keys.size());
    
    std::vector<ValueT> search_results(search_keys.size());
    if (verbose) printf("Starting search of %zu elements...\n", search_keys.size());
    float search_time = hash_table.hash_search(search_keys.data(), search_results.data(), search_keys.size());
    printf("Search Time: %.3f ms (%.3f M elements/s)\n", search_time, double(search_keys.size()) / search_time / 1000.0);
    assert(search_keys.size() == search_results.size());
    
    if (verbose) printf("Starting deletion of %zu elements...\n", remove_keys.size());
    float delete_time = hash_table.hash_delete(remove_keys.data(), remove_keys.size());
    printf("Deletion Time: %.3f ms (%.3f M elements/s)\n", delete_time, double(remove_keys.size()) / delete_time / 1000.0);
    
    double load_factor = hash_table.measureLoadFactor();
    printf("Load factor: %.2f\n", load_factor);
    assert(load_factor >= 0 && load_factor <= 1);
    
    if (verbose) printf("Program completed successfully!\n");
    return 0;
}
