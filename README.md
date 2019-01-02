# pytorch_cpp_samples
PyTorch C++(libtorch) samples

## Usage

Require **libtorch** (https://pytorch.org/cppdocs/installing.html)

```
python export_pt.py
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
./example-app ../model.pt IMAGE_FILE
```

Use GPU
```
./example-app ../model.pt IMAGE_FILE hoge
```
