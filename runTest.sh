###
 # @Description: Run test
 # @Author: Ming Fang
 # @Date: 2021-04-06 15:00:15
 # @LastEditors: Ming Fang
 # @LastEditTime: 2022-02-28 20:43:50
### 

cd ..
mkdir -p build
cd build
cmake ..
## make
cmake --build .
cd ../test_junk
pwd
cp ../bin/main .
./main -i input.json

#./compareOutput.sh
