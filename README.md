# pytorch_cpp_samples
PyTorch C++(libtorch) samples

## Usage

Require **libtorch** (https://pytorch.org/cppdocs/installing.html)

```
python export_pt.py
wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/c2c91c8e767d04621020c30ed31192724b863041/imagenet1000_clsid_to_human.txt
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make
./example-app ../model.pt ../imagenet1000_clsid_to_human.txt IMAGE_FILE
```

Use GPU
```
./example-app ../model.pt ../imagenet1000_clsid_to_human.txt IMAGE_FILE hoge
```
