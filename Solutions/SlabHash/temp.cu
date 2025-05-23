#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::filesystem::path;

static constexpr uint64_t RANDOM_SEED = 42;
static const uint32_t bucket_count = 1000;
static constexpr uint64_t MAX_OPERATIONS = 1e+15;

typedef struct {
  uint32_t key;
  uint32_t value;
} KeyValue;

// Pack key-value into a 64-bit integer
inline uint64_t packKeyValue(uint32_t key, uint32_t val) {
  return (static_cast<uint64_t>(key) << 32) |
         (static_cast<uint32_t>(val) & 0xFFFFFFFF);
}

// Function to unpack a 64-bit integer into two 32-bit integers
inline void unpackKeyValue(uint64_t value, uint32_t& key, uint32_t& val) {
  key = static_cast<uint32_t>(value >> 32);
  val = static_cast<uint32_t>(value & 0xFFFFFFFF);
}

void create_file(path pth, const uint32_t* data, uint64_t size) {
  FILE* fptr = NULL;
  fptr = fopen(pth.string().c_str(), "wb+");
  fwrite(data, sizeof(uint32_t), size, fptr);
  fclose(fptr);
}

/** Read n integer data from file given by pth and fill in the output variable
    data */
void read_data(path pth, uint64_t n, uint32_t* data) {
  FILE* fptr = fopen(pth.string().c_str(), "rb");
  string fname = pth.string();
  if (!fptr) {
    string error_msg = "Unable to open file: " + fname;
    perror(error_msg.c_str());
  }
  int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
  if (freadStatus == 0) {
    string error_string = "Unable to read the file " + fname;
    perror(error_string.c_str());
  }
  fclose(fptr);
}

// These variables may get overwritten after parsing the CLI arguments
/** total number of operations */
uint64_t NUM_OPS = 1e8;
/** percentage of insert queries */
uint64_t INSERT = 100;
/** percentage of delete queries */
uint64_t DELETE = 100;
/** number of iterations */
uint64_t runs = 5;

// List of valid flags and description
void validFlagsDescription() {
  cout << "ops: specify total number of operations\n";
  cout << "rns: the number of iterations\n";
  cout << "add: percentage of insert queries\n";
  cout << "rem: percentage of delete queries\n";
}

// Code snippet to parse command line flags and initialize the variables
int parse_args(char* arg) {
  string s = string(arg);
  string s1;
  uint64_t val;

  try {
    s1 = s.substr(0, 4);
    string s2 = s.substr(5);
    val = stol(s2);
  } catch (...) {
    cout << "Supported: " << std::endl;
    cout << "-*=[], where * is:" << std::endl;
    validFlagsDescription();
    return 1;
  }

  if (s1 == "-ops") {
    NUM_OPS = val;
  } else if (s1 == "-rns") {
    runs = val;
  } else if (s1 == "-add") {
    INSERT = val;
  } else if (s1 == "-rem") {
    DELETE = val;
  } else {
    std::cout << "Unsupported flag:" << s1 << "\n";
    std::cout << "Use the below list flags:\n";
    validFlagsDescription();
    return 1;
  }
  return 0;
}

void batch_insert() {}

void batch_delete() {}

void batch_search() {}

int main(int argc, char* argv[]) {
//   for (int i = 1; i < argc; i++) {
//     int error = parse_args(argv[i]);
//     if (error == 1) {
//       cout << "Argument error, terminating run.\n";
//       exit(EXIT_FAILURE);
//     }
//   }

  uint64_t ADD = NUM_OPS * (INSERT / 100.0);
  uint64_t REM = NUM_OPS * (DELETE / 100.0);
  uint64_t FIND = NUM_OPS - (ADD + REM);

  cout << "NUM OPS: " << NUM_OPS << " ADD: " << ADD << " REM: " << REM
       << " FIND: " << FIND << "\n";

  assert(ADD > 0);

  // auto* h_kvs_insert = new KeyValue[ADD];
  // memset(h_kvs_insert, 0, sizeof(KeyValue) * ADD);
  // auto* h_keys_del = new uint32_t[REM];
  // memset(h_keys_del, 0, sizeof(uint32_t) * REM);
  // auto* h_keys_lookup = new uint32_t[FIND];
  // memset(h_keys_lookup, 0, sizeof(uint32_t) * FIND);
  // Use shared files filled with random numbers
  // path cwd = std::filesystem::path(argv[1]);
  path pathInsert = "/data/vipinpat/trace-files/insert_trace-400e7-100-add-no-dup.bin";
  path pathSearch = "/data/vipinpat/trace-files/search_trace-400e7-30-find-no-dup-no-absent.bin";
  path pathRemove = "/data/vipinpat/trace-files/delete_trace-400e7-10-rem-no-dup-no-absent.bin";
//   path path_insert_values = cwd / "random_values_insert.bin";
//   path path_delete = cwd / "random_keys_delete.bin";
//   path path_search = cwd / "random_keys_search.bin";

  assert(std::filesystem::exists(pathInsert));
  assert(std::filesystem::exists(pathRemove));
  assert(std::filesystem::exists(pathSearch));

  // Read data from file
  auto* tmp_keys_insert = new uint32_t[ADD];
  read_data(pathInsert, ADD, tmp_keys_insert);

  cout << sizeof(tmp_keys_insert);
  // for (int i = 0; i < 100 ; i ++)
    //  std::cout << tmp_keys_insert[i] << " ";
// cout << endl;


  auto h_kvs_insert = new KeyValue[ADD];
  for (int i = 0; i < ADD; i++) {
     h_kvs_insert[i].key = tmp_keys_insert[i];
     h_kvs_insert[i].value = i;
  }
  delete[] tmp_keys_insert;

//   if (REM > 0) {
    auto* tmp_keys_delete = new uint32_t[REM];
    read_data(pathRemove, REM, tmp_keys_delete);
//     for (int i = 0; i < REM; i++) {
//       h_keys_del[i] = tmp_keys_delete[i];
//     }
//     delete[] tmp_keys_delete;
//   }

//   if (FIND > 0) {
    auto* tmp_keys_search = new uint32_t[FIND];
    read_data(pathSearch, FIND, tmp_keys_search);
//     for (int i = 0; i < FIND; i++) {
//       h_keys_lookup[i] = tmp_keys_search[i];
//     }
//     delete[] tmp_keys_search;
//   }

//   // Max limit of the uint32_t: 4,294,967,295
//   std::mt19937 gen(RANDOM_SEED);
//   std::uniform_int_distribution<uint32_t> dist_int(1, NUM_OPS);

//   float total_insert_time = 0.0F;
//   float total_delete_time = 0.0F;
//   float total_search_time = 0.0F;

//   HRTimer start, end;
//   uint32_t del_runs = 0, search_runs = 0;
//   for (uint32_t i = 0; i < runs; i++) {
//     start = HR::now();
//     batch_insert();
//     end = HR::now();
//     float iter_insert_time = duration_cast<milliseconds>(end - start).count();
//     total_insert_time += iter_insert_time;

//     if (REM > 0) {
//       start = HR::now();
//       batch_delete();
//       end = HR::now();
//       float iter_delete_time = duration_cast<milliseconds>(end - start).count();
//       del_runs++;
//       total_delete_time += iter_delete_time;
//     }

//     if (FIND > 0) {
//       start = HR::now();
//       batch_search();
//       end = HR::now();
//       float iter_search_time = duration_cast<milliseconds>(end - start).count();
//       search_runs++;
//       total_search_time += iter_search_time;
//     }
//   }

//   cout << "Time taken by insert kernel (ms): " << total_insert_time / runs
//        << "\n";
//   if (del_runs > 0) {
//     cout << "Time taken by delete kernel (ms): " << total_delete_time / del_runs
//          << "\n";
//   }
//   if (search_runs > 0) {
//     cout << "Time taken by search kernel (ms): "
//          << total_search_time / search_runs << "\n";
//   }

//   return EXIT_SUCCESS;
}
