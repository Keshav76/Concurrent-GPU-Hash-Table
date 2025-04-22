nvcc -ccbin /usr/bin/gcc-10 -std=c++17 exp.cu -o output -lcudart -lstdc++ -arch=sm_60 --expt-relaxed-constexpr -lm -w

./output --size=100000000 --ins=1 --lkp=0.4 --rem=0.1 --insert_path=insert_trace-400e7-100-add-no-dup.bin --search_path=search_trace-400e7-30-find-no-dup-no-absent.bin --delete_path=delete_trace-400e7-10-rem-no-dup-no-absent.bin > report.txt