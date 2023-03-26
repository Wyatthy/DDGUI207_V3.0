INCLUDEPATH += \
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include" \
  H:/Linux_DATA/jwk/DD_GUI_ENVs/TensorRT-7.2.3.4/include \
  H:/WIN_LIB/MATLAB/R2022a/extern/include \
  D:/lyh/GUI207_V2.0/lib/TRANSFER \
  H:/WIN_LIB/Anaconda3/include \
  H:/WIN_LIB/Anaconda3/Lib/site-packages/numpy/core/include/numpy \
  H:/Linux_DATA/jwk/DD_GUI_ENVs/OpenCV4.5.4/opencv/build/include

LIBS += \
  -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64" \
  -lcudart \
  -lcuda \
  -lcudadevrt \
  -LH:/Linux_DATA/jwk/DD_GUI_ENVs/TensorRT-7.2.3.4/lib \
  -lnvinfer \
  -LH:/WIN_LIB/MATLAB/R2022a/extern/lib/win64/microsoft \
  -llibmat \
  -llibmx \
  -llibmex \
  -llibeng \
  -lmclmcr \
  -lmclmcrrt \
  -LD:/lyh/GUI207_V2.0/build/lib/TRANSFER \
  -lToHrrp \
  -LH:/WIN_LIB/Anaconda3/libs \
  -lpython39 \
  -LH:/WIN_LIB/Anaconda3/Lib/site-packages/numpy/core/lib \
  -lnpymath \
  -LH:/Linux_DATA/jwk/DD_GUI_ENVs/OpenCV4.5.4/opencv/build/x64/vc15/lib \
  -lopencv_world454

