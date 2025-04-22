#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <warpcore/single_value_hash_table.cuh>
#include <helpers/timers.cuh>
float startTimer(cudaEvent_t &start, cudaEvent_t &stop){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    return 0.0f;
}
float stopTimer(cudaEvent_t start, cudaEvent_t stop){
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}
int main(int argc, char* argv[])
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <insert_file> <search_file> <erase_file> <n>\n";
        return 1;
    }

    const std::string insert_file = argv[1];
    const std::string search_file = argv[2];
    const std::string erase_file = argv[3];
    const uint64_t total_size = std::stoull(argv[4]);

    using key_t   = std::uint32_t;
    using value_t = std::uint32_t;

    using namespace warpcore;
    using hash_table_t = SingleValueHashTable<key_t, value_t>;
    printf("------------------------Warpcore------------------------\n");
    std::cout << "Total elements: " << total_size << std::endl;

    // Initialize random number generator for values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<value_t> dis(1, 1000000);

    // ------------------------- INITIALIZE ALL ARRAYS -------------------------
    // Host arrays for full dataset
    std::vector<key_t> insert_keys(total_size);
    std::vector<value_t> insert_values(total_size);
    std::vector<key_t> search_keys(total_size);
    std::vector<key_t> erase_keys(total_size);

    // Read all insert keys from file and generate random values
    std::ifstream insert_in(insert_file, std::ios::binary);
    if (!insert_in) {
        std::cerr << "Error opening insert binary file.\n";
        return 1;
    }
    for (uint64_t i = 0; i < total_size; ++i) {
        insert_in.read(reinterpret_cast<char*>(&insert_keys[i]), sizeof(uint32_t));
        if (insert_in.gcount() != sizeof(uint32_t)) {
            std::cerr << "Error reading key from insert file.\n";
            return 1;
        }
        insert_values[i] = dis(gen);
    }
    insert_in.close();

    // Read all search keys from file
    std::ifstream search_in(search_file, std::ios::binary);
    if (!search_in) {
        std::cerr << "Error opening search binary file.\n";
        return 1;
    }
    for (uint64_t i = 0; i < total_size; ++i) {
        search_in.read(reinterpret_cast<char*>(&search_keys[i]), sizeof(uint32_t));
        if (search_in.gcount() != sizeof(uint32_t)) {
            std::cerr << "Error reading key from search file.\n";
            return 1;
        }
    }
    search_in.close();

    // Read all erase keys from file
    std::ifstream erase_in(erase_file, std::ios::binary);
    if (!erase_in) {
        std::cerr << "Error opening erase binary file.\n";
        return 1;
    }
    for (uint64_t i = 0; i < total_size; ++i) {
        erase_in.read(reinterpret_cast<char*>(&erase_keys[i]), sizeof(uint32_t));
        if (erase_in.gcount() != sizeof(uint32_t)) {
            std::cerr << "Error reading key from erase file.\n";
            return 1;
        }
    }
    erase_in.close();

    // ------------------------- SETUP GPU RESOURCES -------------------------
    // Calculate hash table capacity
    const float load = 0.85;
    const uint64_t capacity = total_size / load;
    cudaEvent_t start, stop;


    // Create hash table
    hash_table_t hash_table(capacity); CUERR
    // std::cout << "Hash table capacity: " << hash_table.capacity() << std::endl;

    // Allocate and copy all data to device at once
    key_t* insert_keys_d; cudaMalloc(&insert_keys_d, sizeof(key_t) * total_size); CUERR
    value_t* insert_values_d; cudaMalloc(&insert_values_d, sizeof(value_t) * total_size); CUERR
    key_t* search_keys_d; cudaMalloc(&search_keys_d, sizeof(key_t) * total_size); CUERR
    key_t* erase_keys_d; cudaMalloc(&erase_keys_d, sizeof(key_t) * total_size); CUERR
    value_t* results_d; cudaMalloc(&results_d, sizeof(value_t) * total_size); CUERR

    cudaMemcpy(insert_keys_d, insert_keys.data(), sizeof(key_t) * total_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(insert_values_d, insert_values.data(), sizeof(value_t) * total_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(search_keys_d, search_keys.data(), sizeof(key_t) * total_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(erase_keys_d, erase_keys.data(), sizeof(key_t) * total_size, cudaMemcpyHostToDevice); CUERR

    // ------------------------- INSERT OPERATION -------------------------
    startTimer(start, stop);
        hash_table.insert(insert_keys_d, 
                         insert_values_d, 
                         total_size);
        cudaDeviceSynchronize(); CUERR

    float insert_end = stopTimer(start, stop);
    std::cout << "Insert Time = " << insert_end << " ms | Throughput = "  << total_size * 0.001 / insert_end << " M ops/sec\n";
    // ------------------------- RETRIEVE OPERATION -------------------------
    startTimer(start, stop);
        hash_table.retrieve(search_keys_d, 
                          total_size, 
                          results_d);
        cudaDeviceSynchronize(); CUERR


    float retrieve_end = stopTimer(start, stop);

    // Copy all results back at once
    std::vector<value_t> results_h(total_size);
    cudaMemcpy(results_h.data(), results_d, sizeof(value_t) * total_size, cudaMemcpyDeviceToHost); CUERR
    std::cout << "Search Time = " << retrieve_end << " ms | Throughput = "  << total_size * 0.001 / retrieve_end << " M ops/sec\n";

    // ------------------------- ERASE OPERATION -------------------------
    startTimer(start, stop);
        hash_table.erase(erase_keys_d , total_size);
        cudaDeviceSynchronize(); CUERR
    float erase_end = stopTimer(start, stop);
    std::cout << "Delete Time = " << erase_end << " ms | Throughput = "  << total_size * 0.001 / erase_end << " M ops/sec\n";

    // ------------------------- CLEANUP -------------------------
    // Free device memory
    cudaFree(insert_keys_d);
    cudaFree(insert_values_d);
    cudaFree(search_keys_d);
    cudaFree(erase_keys_d);
    cudaFree(results_d);

    cudaDeviceSynchronize(); CUERR
}