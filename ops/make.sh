set -x 
set -e

cd query_depth_point
python3 setup.py build_ext --inplace
cd ..

cd pybind11
python3 setup.py build_ext --inplace

# include=`python -m pybind11 --includes`
# g++ -std=c++11 -shared -o box_ops_cc.so box_ops.cc -fPIC -O3 ${include}