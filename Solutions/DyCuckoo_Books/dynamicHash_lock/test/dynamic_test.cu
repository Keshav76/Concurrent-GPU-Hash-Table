#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../tools/gputimer.h"
#include "../data/data_layout.cuh"
#include "../core/dynamic_cuckoo.cuh"
namespace ch = cuckoo_helpers;
using namespace std;


class DynamicTest {
public:
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    static constexpr uint32_t val_lens = DataLayout<>::val_lens;

    int r = 2;
    int batch_size = 100000;  //smaller batch size: 2e4 4e4 6e4 8e4 10e4
    double lower_bound = 0.5;  //lower bound: 0.3 0.35 0.4 0.45 0.5
    double upper_bound = 0.85; //upper bound: 0.7 0.75 0.8 0.85 0.9
    int pool_len = 0;
    key_t *keys_pool_d;          // For insert keys
    key_t *search_keys_pool_d;   // For search keys
    key_t *delete_keys_pool_d;   // For delete keys
    value_t *value_pool_d, *check_pool_d;
    double init_fill_factor = 0.85;
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
    static key_t *read_data(char *file_name, int data_len) {
        FILE *fid;
        fid = fopen(file_name, "rb");
        key_t *pos = (key_t *) malloc(sizeof(key_t) * data_len);
        if (fid == NULL) {
            printf("file not found.\n");
            return pos;
        }
        fread(pos, sizeof(unsigned int), data_len, fid);
        fclose(fid);
        return pos;
    }
    static value_t *read_data_val(char *file_name, int data_len) {
        FILE *fid;
        fid = fopen(file_name, "rb");
        value_t *pos = (value_t *) malloc(sizeof(value_t) * data_len);
        if (fid == NULL) {
            printf("file not found.\n");
            return pos;
        }
        fread(pos, sizeof(unsigned int), data_len, fid);
        fclose(fid);
        return pos;
    }
    
    void batch_check(value_t *check_pool_d, int32_t single_batch_size, uint32_t offset) {
        uint32_t error_cnt = 0;
        value_t *check_pool_h = new value_t[single_batch_size];
        cudaMemcpy(check_pool_h, check_pool_d + offset, sizeof(value_t) * single_batch_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < single_batch_size; i++) {
            for(int j = 0; j < val_lens; j++){
                if(check_pool_h[i].data[j] != i + 5 + offset){
                    ++error_cnt;
                    break;
                }
            }
        }
        if (error_cnt != 0) {
            printf("num error:%d \n", error_cnt);
        } else {
            printf("batch check ok\n");
        }
        delete[] check_pool_h;
    }

    // void batch_test() {
    //     DynamicCuckoo<512, 512> dy_cuckoo((uint32_t)batch_size * 10 / init_fill_factor, batch_size, lower_bound, upper_bound);
    //     int32_t batch_num = pool_len / batch_size;
    //     printf("pool_len:::::::::::::::: %d\n", pool_len);
    //     int32_t batch_round = batch_num / 10;
    //     cudaEvent_t start, stop;
    //     // for (int repeat = 0; repeat < 10; repeat++) {
    //         float insert_time =0, search_time = 0, delete_time = 0;
    //         for (int32_t batch_round_ptr = 0; batch_round_ptr < batch_round; ++batch_round_ptr) {
    //             int batch_ptr = batch_round_ptr * 10;
    //             startTimer(start, stop);
    //             for (int j = 0; j < 10; j++) {
    //                 dy_cuckoo.batch_insert(keys_pool_d + (batch_ptr + j) * batch_size,
    //                                      value_pool_d + (batch_ptr + j) * batch_size, batch_size);
    //             }
    //             insert_time += stopTimer(start, stop);
    //             startTimer(start, stop);
    //             for (int j = 0; j < 10; j++) {
    //                 dy_cuckoo.batch_search(search_keys_pool_d + (batch_ptr + j) * batch_size,
    //                                      check_pool_d + (batch_ptr + j) * batch_size, batch_size);
    //             }
    //             search_time += stopTimer(start, stop);
    //             startTimer(start, stop);
    //             for (int j = 0; j < 10; j++) {
    //                 dy_cuckoo.batch_delete(delete_keys_pool_d + (batch_ptr + j) * batch_size, nullptr, batch_size);
    //             }
    //             delete_time += stopTimer(start, stop);
                
    //         }
    //         double insert_throughput = (10 * batch_size) / ((insert_time / batch_round) * 1e-3);
    //         double search_throughput = (10 * batch_size) / ((search_time / batch_round) * 1e-3);
    //         double delete_throughput = (10 * batch_size) / (delete_time / batch_round * 1e-3);
            
    //         printf("Insert Time = %.3f ms | Throughput = %.2lf M ops/sec\n", 
    //                insert_time / batch_round, insert_throughput / 1e6);
    //         printf("Search Time = %.3f ms | Throughput = %.2lf M ops/sec\n", 
    //             search_time / batch_round, search_throughput / 1e6);
    //         printf("Delete Time = %.3f ms | Throughput = %.2lf M ops/sec\n", 
    //             delete_time / batch_round, delete_throughput / 1e6);
    //     // }

    // }
        void batch_test() {
        DynamicCuckoo<512, 512> dy_cuckoo((uint32_t)batch_size * 10 / init_fill_factor, batch_size, lower_bound, upper_bound);
        int32_t batch_num = pool_len / batch_size;
        printf("------------------------DyCuckoo------------------------\n");
        printf("Total elements: %d\n", pool_len);
        int32_t batch_round = batch_num / 10;
        cudaEvent_t start, stop;
        float insert_time =0, search_time = 0, delete_time = 0;
        startTimer(start, stop);
        dy_cuckoo.batch_insert(keys_pool_d, value_pool_d, pool_len);
        cudaDeviceSynchronize();
        insert_time = stopTimer(start, stop);
        
        startTimer(start, stop);
        dy_cuckoo.batch_search(search_keys_pool_d, check_pool_d , pool_len);
        cudaDeviceSynchronize();
        search_time = stopTimer(start, stop);
        
        startTimer(start, stop);
        dy_cuckoo.batch_delete(delete_keys_pool_d, nullptr, pool_len);
        cudaDeviceSynchronize();
        delete_time = stopTimer(start, stop);
        
        double insert_throughput = pool_len * 0.001/ (insert_time);
        double search_throughput = pool_len * 0.001/ (search_time);
        double delete_throughput = pool_len * 0.001 / (delete_time);
        
        printf("Insert Time = %.3f ms | Throughput = %.2lf M ops/sec\n", insert_time, insert_throughput );
        printf("Search Time = %.3f ms | Throughput = %.2lf M ops/sec\n", search_time , search_throughput);
        printf("Delete Time = %.3f ms | Throughput = %.2lf M ops/sec\n", delete_time , delete_throughput);

    }
};


int main(int argc, char** argv) {
    using test_t = DynamicTest;
    if (argc < 10)
    {
        cout << "Usage: " << argv[0] << " insert_file search_file delete_file pool_len r batch_size lower_bound upper_bound init_fill_factor\n";
        cout << "para error\n" << endl;
        return -1;
    }

    test_t dy_test;
    char* insert_file_name = argv[1];
    char* search_file_name = argv[2];
    char* delete_file_name = argv[3];
    int pool_len = atoi(argv[4]);
    dy_test.pool_len = pool_len;
    dy_test.r = atoi(argv[5]);
    dy_test.batch_size = atoi(argv[6]) / 10;
    dy_test.lower_bound = atof(argv[7]);
    dy_test.upper_bound = atof(argv[8]);
    dy_test.init_fill_factor = atof(argv[9]);

    // Read keys from three different files
    test_t::key_t* insert_keys_h = test_t::read_data(insert_file_name, pool_len);

    test_t::key_t* search_keys_h = test_t::read_data(insert_file_name, pool_len);

    test_t::key_t* delete_keys_h = test_t::read_data(insert_file_name, pool_len);


    test_t::value_t *values_h = test_t::read_data_val(search_file_name, pool_len);
    test_t::value_t *check_h = new test_t::value_t [pool_len];
    
    for(int i = 0; i < pool_len; i++){
        for(int j = 0; j < DataLayout<>::val_lens; j++){
            // values_h[i].data[j] = i + 5;
            check_h[i].data[j] = 0;
        }
    }

    // Allocate and copy all three key sets to device
    cudaMalloc((void**)&(dy_test.keys_pool_d), sizeof(test_t::key_t) * pool_len);
    cudaMalloc((void**)&(dy_test.search_keys_pool_d), sizeof(test_t::key_t) * pool_len);
    cudaMalloc((void**)&(dy_test.delete_keys_pool_d), sizeof(test_t::key_t) * pool_len);
    cudaMalloc((void**)&(dy_test.value_pool_d), sizeof(test_t::value_t) * pool_len);
    cudaMalloc((void**)&(dy_test.check_pool_d), sizeof(test_t::value_t) * pool_len);

    cudaMemcpy(dy_test.keys_pool_d, insert_keys_h, sizeof(test_t::key_t) * pool_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dy_test.search_keys_pool_d, search_keys_h, sizeof(test_t::key_t) * pool_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dy_test.delete_keys_pool_d, delete_keys_h, sizeof(test_t::key_t) * pool_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dy_test.value_pool_d, values_h, sizeof(test_t::value_t) * pool_len, cudaMemcpyHostToDevice);

    dy_test.batch_test();

    delete []insert_keys_h;
    delete []search_keys_h;
    delete []delete_keys_h;
    delete []values_h;
    delete []check_h;
    return 0;
}
