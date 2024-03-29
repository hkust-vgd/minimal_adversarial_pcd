TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda-11.6/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance.so -shared -fPIC  -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework  -I /usr/local/cuda-11.6/include -lcudart -L /usr/local/cuda-11.6/lib64/ -O2