![header](docs/assets/LogoHeader.png)

# Floor Plan Analysis Hub: _Swin Transformer + Mask R-CNN_

Complete system for detecting and segmenting rooms in floor plans using Deep Learning.

> To use other implemented architectures, see the [Implementation Index](https://github.com/BenjaSar/floorplan-classifier/blob/main/README.md) on the main branch.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)
![Django](https://img.shields.io/badge/Django-5.2.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Accurate detection** of 14 different room types
- 🎨 **Pixel-level mask segmentation**
- 📊 **Automatic area calculation** in square meters
- 🌐 **Django web interface** with real-time visualization
- 🔄 **Synthetic dataset** of 500 floor plans generated automatically
- 🚀 **Modern architecture**: Swin Transformer + Mask R-CNN

## 🏗️ Architecture

```
Swin Transformer (Backbone)
  ↓
Feature Pyramid Network
  ↓
Region Proposal Network
  ↓
ROI Align + Box/Mask Heads
  ↓
Detections + Masks + Areas
```

## 📦 Quick Installation

```bash
# Clone repository
git clone https://github.com/BenjaSar/floorplan-classifier.git
cd floorplan-classifier

# Switch to desired model branch
git checkout swin_maskrcnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset (500 plans)
python utils/synthetic_data_generator.py

# Create initialized model weights
python create_pretrained_weights.py

# Start Django server
python manage.py runserver 8080
```

### If you want to test a different model
```bash
# Option for going back to model's index
git checkout main
```

## 🚀 Usage

1. **Open browser**: http://127.0.0.1:8080/

2. **Upload plan**: Click "Upload Image" and select a floor plan

3. **View results**:
   - Detected rooms with bounding boxes
   - Colored segmentation masks
   - Detailed table with areas in m²
   - Global statistics

## 🏷️ Supported Room Types

- 🛏️ Bedroom
- 🍳 Kitchen
- 🛋️ Living Room
- 🚿 Bathroom
- 🍽️ Dining Room
- 🚪 Corridor
- 🌅 Balcony
- 📦 Storage
- 🚗 Garage
- 🧺 Laundry
- 💼 Office
- 🛌 Guest Room
- 🔧 Utility
- ❓ Other

## 📂 Project Structure

```
floorplan-classifier/
├── src/
│   └── models/
│       └── swin_maskrcnn.py       # Main model
├── utils/
│   ├── synthetic_data_generator.py # Data generator
│   ├── visualization.py            # Visualization
│   └── area_calculator.py          # Area calculation
├── detector/
│   ├── views.py                    # Django logic
│   └── templates/                  # HTML templates
├── webapp/
│   ├── settings.py                 # Configuration
│   └── urls.py                     # URLs
├── checkpoints/                    # Model weights (not included)
├── data/                           # Dataset (not included)
├── create_pretrained_weights.py    # Script for weights
└── manage.py                       # Django CLI
```

## 🎓 Train the Model (Optional)

```bash
# Fast training (demo)
python train_fast.py

# Full training
python train.py --epochs 100 --batch-size 4
```

## 🔧 Technologies

- **Backend**: Django 5.2.8
- **Deep Learning**: PyTorch 2.9.1
- **Computer Vision**: OpenCV, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Data Science**: NumPy, Pandas

## 📊 Dataset

- **Synthetic**: 500 generated floor plans (400 train, 50 val, 50 test)
- **Format**: COCO (JSON annotations)
- **Resolution**: 512x512 pixels
- **Annotations**: Perfect (no human errors)

## 🎯 Model Metrics

- **Parameters**: ~100M
- **Size**: 138 MB
- **Input**: 512x512 RGB
- **Output**: Boxes + Masks + Labels + Scores

## Related Papers
- **CubiCasa5K:** [“CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis”](https://arxiv.org/abs/1904.01920)
- **DeiT:** [“Training data-efficient image transformers”](https://arxiv.org/abs/2012.12877)
- **Swin Transformer:** [“Swin Transformer: Hierarchical Vision Transformer using Shifted Windows”](https://arxiv.org/abs/2103.14030)
- **Mask R-CNN:** [“Mask R-CNN”](https://arxiv.org/abs/1703.06870)

---
## 📖 Documentation

- [PROYECTO_COMPLETO.md](docs/PROYECTO_COMPLETO.md) - Full documentation in Spanish
- [EMPEZAR_AQUI.md](docs/EMPEZAR_AQUI.md) - Quick start guide
- [INFORME_TECNICO.md](docs/INFORME_TECNICO.md) - Detailed technical report

## 🐛 Fixed Issues

- ✅ CUDA compatibility (forced to CPU)
- ✅ Boolean index error in masks
- ✅ Array synchronization in visualization
- ✅ Correct handling of dimensions (N, 1, H, W)
- ✅ Background filtering before processing

## 🚀 Planned Improvements

- [ ] Training with real dataset (CubiCasa5K)
- [ ] GPU optimization
- [ ] REST API for integration
- [ ] Export to ONNX/TensorRT
- [ ] Advanced data augmentation
- [ ] Evaluation metrics (mAP, IoU)

---
## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{floorplan_classifier,
  title={Floor Plan Vision Transformer Classifier},
  author={Grupo 3 VpC},
  year={2025},
  url={https://github.com/BenjaSar/floorplan-classifier}
}
```

## 👤 Author

**Jorge Cuenca** ([@Jorgecuenca1](https://github.com/Jorgecuenca1))

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) for the dataset
- [Common Objects in Context (COCO)](https://cocodataset.org/)
- [Swin Transformer Architecture (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/swin-transformer-supports-3-billion-parameter-vision-models-that-can-train-with-higher-resolution-images-for-greater-task-applicability/)
- [Mask R-CNN Framework (Facebook AI Research)](https://github.com/facebookresearch/maskrcnn-benchmark)
- [OpenCV](https://opencv.org/) for image processing
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [MLflow](https://mlflow.org/) for experiment tracking

## 📞 Contact

For questions or issues:

- **GitHub Issues**: [Create an issue](https://github.com/BenjaSar/floorplan-classifier/issues)

---

**Made with ❤️ for the architecture and computer vision communities**

*Last Updated: November 23, 2025*

![footer](docs/assets/LogoFooter.png)