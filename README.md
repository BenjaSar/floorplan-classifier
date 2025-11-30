![header](docs/assets/LogoHeader.png)

# Floor Plan Analysis Hub: _Swin Transformer + Mask R-CNN_

Sistema completo de detección y segmentación de habitaciones en planos de planta usando Deep Learning.

> To use other implemented architectures, see the [Implementation Index](https://github.com/BenjaSar/floorplan-classifier/blob/main/README.md) on the main branch.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)
![Django](https://img.shields.io/badge/Django-5.2.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Características

- 🎯 **Detección precisa** de 14 tipos de habitaciones diferentes
- 🎨 **Segmentación por máscaras** a nivel de píxel
- 📊 **Cálculo automático de áreas** en metros cuadrados
- 🌐 **Interfaz web Django** con visualización en tiempo real
- 🔄 **Dataset sintético** de 500 planos generados automáticamente
- 🚀 **Arquitectura moderna**: Swin Transformer + Mask R-CNN

## 🏗️ Arquitectura

```
Swin Transformer (Backbone)
    ↓
Feature Pyramid Network
    ↓
Region Proposal Network
    ↓
ROI Align + Box/Mask Heads
    ↓
Detecciones + Máscaras + Áreas
```

## 📦 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/Jorgecuenca1/floorplan-classifier.git
cd floorplan-classifier

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Generar dataset sintético (500 planos)
python utils/synthetic_data_generator.py

# Crear pesos inicializados del modelo
python create_pretrained_weights.py

# Iniciar servidor Django
python manage.py runserver 8080
```

## 🚀 Uso

1. **Abrir navegador**: http://127.0.0.1:8080/

2. **Subir plano**: Click en "Subir Imagen" y selecciona un plano de planta

3. **Ver resultados**:
   - Habitaciones detectadas con bounding boxes
   - Máscaras de segmentación coloreadas
   - Tabla detallada con áreas en m²
   - Estadísticas globales

## 🏷️ Tipos de Habitaciones Soportadas

- 🛏️ Bedroom (Dormitorio)
- 🍳 Kitchen (Cocina)
- 🛋️ Living Room (Sala)
- 🚿 Bathroom (Baño)
- 🍽️ Dining Room (Comedor)
- 🚪 Corridor (Pasillo)
- 🌅 Balcony (Balcón)
- 📦 Storage (Almacenamiento)
- 🚗 Garage (Garage)
- 🧺 Laundry (Lavandería)
- 💼 Office (Oficina)
- 🛌 Guest Room (Cuarto de Huéspedes)
- 🔧 Utility (Utilidad)
- ❓ Other (Otros)

## 📂 Estructura del Proyecto

```
floorplan-classifier/
├── src/
│   └── models/
│       └── swin_maskrcnn.py       # Modelo principal
├── utils/
│   ├── synthetic_data_generator.py # Generador de datos
│   ├── visualization.py            # Visualización
│   └── area_calculator.py          # Cálculo de áreas
├── detector/
│   ├── views.py                    # Lógica Django
│   └── templates/                  # Templates HTML
├── webapp/
│   ├── settings.py                 # Configuración
│   └── urls.py                     # URLs
├── checkpoints/                    # Pesos del modelo (no incluido)
├── data/                          # Dataset (no incluido)
├── create_pretrained_weights.py   # Script para pesos
└── manage.py                      # Django CLI
```

## 🎓 Entrenar el Modelo (Opcional)

```bash
# Entrenamiento rápido (demo)
python train_fast.py

# Entrenamiento completo
python train.py --epochs 100 --batch-size 4
```

## 🔧 Tecnologías

- **Backend**: Django 5.2.8
- **Deep Learning**: PyTorch 2.9.1
- **Computer Vision**: OpenCV, Pillow
- **Visualización**: Matplotlib, Seaborn
- **Data Science**: NumPy, Pandas

## 📊 Dataset

- **Sintético**: 500 planos generados (400 train, 50 val, 50 test)
- **Formato**: COCO (anotaciones JSON)
- **Resolución**: 512x512 píxeles
- **Anotaciones**: Perfectas (sin errores humanos)

## 🎯 Métricas del Modelo

- **Parámetros**: ~100M
- **Tamaño**: 138 MB
- **Input**: 512x512 RGB
- **Output**: Boxes + Máscaras + Labels + Scores

## Related Papers
- **CubiCasa5K:** [«CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis»](https://arxiv.org/abs/1904.01920)
- **DeiT:** [«Training data-efficient image transformers»](https://arxiv.org/abs/2012.12877)
- **Swin Transformer:** [«Swin Transformer: Hierarchical Vision Transformer using Shifted Windows»](https://arxiv.org/abs/2103.14030)
- **Mask R-CNN:** [«Mask R-CNN»](https://arxiv.org/abs/1703.06870)

---
## 📖 Documentación

- [PROYECTO_COMPLETO.md](docs/PROYECTO_COMPLETO.md) - Documentación completa en español
- [EMPEZAR_AQUI.md](docs/EMPEZAR_AQUI.md) - Guía de inicio rápido
- [INFORME_TECNICO.md](docs/INFORME_TECNICO.md) - Análisis técnico detallado

## 🐛 Problemas Resueltos

- ✅ Compatibilidad CUDA (forzado a CPU)
- ✅ Error de boolean index en máscaras
- ✅ Sincronización de arrays en visualización
- ✅ Manejo correcto de dimensiones (N, 1, H, W)
- ✅ Filtrado de Background antes de procesamiento

## 🚀 Próximas Mejoras

- [ ] Entrenamiento con dataset real (CubiCasa5K)
- [ ] Optimización para GPU
- [ ] API REST para integración
- [ ] Exportación a ONNX/TensorRT
- [ ] Data augmentation avanzada
- [ ] Métricas de evaluación (mAP, IoU)

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

## 👤 Autor

**Jorge Cuenca** ([@Jorgecuenca1](https://github.com/Jorgecuenca1))

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) for the dataset
- [Common Objects in Context (COCO)](https://cocodataset.org/)
- [Swin Transformer Architechture (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/swin-transformer-supports-3-billion-parameter-vision-models-that-can-train-with-higher-resolution-images-for-greater-task-applicability/)
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