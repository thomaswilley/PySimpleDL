# SimpleDL

A simple L-layer machine learning library with minimal dependencies and optional interoperability with ONNX.

- Design, train, load, and save ML/DL models (Including to [ONNX][0]!)
- Easy to embed in edge or cloud services for inferencing
- Simple and small, good for learning

Installation:
```bash
$ bash pip install git+https://github.com/thomaswilley/PySimpleDL.git
```

Usage: Check out example/ for a few examples of usage.

```python
from simpledl.DLTrainer import DLTrainer
from simpledl.ModelManager import ModelManager
```

## Author

**(c) Thomas Willey**
- <https://github.com/thomaswilley>
- <https://twitter.com/thomaswilley>

Additional copyrights included within source.

## License

Open sourced under the [BSD license](LICENSE).

[0]: https://onnx.ai
