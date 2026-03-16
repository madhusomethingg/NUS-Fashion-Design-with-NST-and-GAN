# 👗 Fashion Design Using Neural Style Transfer and GANs
### Synthesizing Custom Clothing Designs by Combining User Style Preferences with Deep Learning

> A deep learning pipeline that generates new clothing designs by applying Neural Style Transfer and a custom-built GAN to synthesize art styles onto existing garment images, paired with an interactive Flask app powered by the GrabCut algorithm for precise foreground extraction. Developed as a research intern at the National University of Singapore (NUS) in collaboration with Hewlett Packard Enterprise (HPE), and presented to professors at the NUS School of Computing and Senior Directors at HPE.

---

## 📌 Overview

Fashion design is one of the last creative industries to be touched by artificial intelligence. Most design tools assist; they don't generate. This project asks whether deep learning can go further: can a model learn a user's style preferences from a small set of inputs and synthesize entirely new clothing designs from them?

This project was developed during a research internship at the National University of Singapore (NUS), in collaboration with Hewlett Packard Enterprise (HPE). The work was presented to professors at the NUS School of Computing and Senior Directors at HPE as part of the internship deliverable.

The pipeline combines two distinct approaches: Neural Style Transfer, which blends the visual style of an artwork onto a content garment image, and a GAN model trained on art data that generates novel design textures from noise. To preserve the clothing's structure during stylization, the GrabCut algorithm is used to isolate the garment outline, ensuring the generated design wraps the clothing shape and not the background.

| Goal | Approach |
|------|----------|
| Generate new clothing designs from user style preferences | Neural Style Transfer (VGG-19) applied to garment and art image pairs |
| Synthesize novel art-inspired textures for fashion | Custom GAN trained on resized art dataset |
| Preserve garment structure during stylization | GrabCut-based foreground segmentation via interactive Flask app |
| Enable user-driven design customization | Web interface for rectangle-based and stroke-refined segmentation |

---

## 📂 Dataset

**Art Images Dataset — Resized for GAN Training**

- **File:** `art_data.npy` (preprocessed numpy array)
- **Source:** Custom art image collection, resized to 128×128 RGB
- **Link:** [Google Colab](https://colab.research.google.com/drive/1Scm-wgaB3O4tUFNVj9JnaYKuEUSLpi_e?usp=sharing)
- **Preprocessing:** Images resized using PIL, normalized to `[-1, 1]` range via `image_resizer.py`
- **GAN Input:** Flattened into `(-1, 128, 128, 3)` array and saved as `.npy` for training

### Data Pipeline

| Step | Description |
|------|-------------|
| **Image resizing** | All art images resized to 128×128 RGB using `PIL.Image.ANTIALIAS` |
| **Normalization** | Pixel values normalized to `[-1, 1]` for GAN training stability |
| **Storage** | Saved as `art_data.npy` for fast loading during training |

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-------------------|
| Deep Learning | `Keras`, `TensorFlow` |
| GAN Architecture | `Conv2D`, `UpSampling2D`, `LeakyReLU`, `BatchNormalization` |
| Image Segmentation | `OpenCV` (GrabCut algorithm) |
| Web Application | `Flask` |
| Image Processing | `PIL`, `numpy`, `cv2` |
| Training Environment | Google Colaboratory |

---

## 🗂️ Repository Structure

```
📦 fashion-design-with-neural-style-transfer-and-gans
 ┣ 📜 implementation.ipynb      # GAN model: generator, discriminator, training loop
 ┣ 📜 image_resizer.py          # Resizes and preprocesses art images into art_data.npy
 ┣ 📜 grabcut.py                # GrabCut segmentation logic (rect and drawing refinement)
 ┣ 📜 __init__.py               # Flask app: routes for GrabCut and refinement
 ┣ 📜 run_app.ipynb             # Launches the Flask server
 ┣ 📂 static
 ┃ ┣ 📂 images                  # Input garment images organized by project
 ┃ ┗ 📂 js
 ┃   ┗ 📜 grabcut_demo.js       # Frontend canvas logic for drawing and interaction
 ┣ 📂 templates
 ┃ ┗ 📜 grabcut_demo.html       # Main UI template
 ┗ 📜 README.md
```

---

## 🔬 Methodology

### 1. Image Preprocessing
- Art images collected and resized to 128×128 RGB using `image_resizer.py`
- Pixel values normalized to the `[-1, 1]` range required by the GAN's tanh output activation
- Processed images saved as `art_data.npy` for efficient loading during training

### 2. GAN Architecture

A custom Deep Convolutional GAN (DCGAN) was built from scratch using Keras. The generator learns to synthesize 128×128 art-style images from 100-dimensional noise vectors; the discriminator learns to distinguish real art images from generated ones.

**Generator** upsamples noise into a full image via progressive convolutional blocks:

```python
def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    # Progressive upsampling blocks (x5) with BatchNormalization
    for i in range(GENERATE_RES):
        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
```

**Discriminator** classifies images as real or fake through progressive downsampling:
- 5 convolutional layers (32 → 64 → 128 → 256 → 512 filters)
- LeakyReLU activations, BatchNormalization, and Dropout for training stability
- Binary cross-entropy loss with Adam optimizer (`lr=1.5e-4`)

> **Key Insight:** The generator and discriminator are trained adversarially. The generator improves by fooling the discriminator, while the discriminator improves by catching fakes. Over 5000 epochs with batch size 64, this produces progressively more realistic art-style textures.

### 3. GrabCut Segmentation

To preserve the garment's shape during Neural Style Transfer, the GrabCut algorithm is used to isolate the clothing from its background, producing a clean alpha-masked PNG.

The Flask web app provides two levels of segmentation control:

| Step | User Action | Backend Process |
|------|-------------|-----------------|
| **Initial cut** | Draw a rectangle around the garment | `grabcut_rect()` initializes GrabCut with `GC_INIT_WITH_RECT` |
| **Refinement** | Draw strokes to mark background and foreground | `grabcut_drawing()` reinitializes with `GC_INIT_WITH_MASK` |

The output is a transparent-background PNG (RGBA) with only the garment retained, saved locally and returned to the frontend.

### 4. Neural Style Transfer

The isolated garment image (content) is combined with a user-selected artwork (style) using VGG-19-based Neural Style Transfer, generating a new clothing design that carries the texture and color palette of the chosen artwork while preserving the garment's original structure.

---

## 📊 Key Results

- GAN successfully synthesized novel art-style textures after training on 128×128 art images over 5000 epochs
- GrabCut-based segmentation cleanly isolated garment foregrounds, enabling structure-preserving style transfer
- VGG-19 NST produced visually coherent clothing designs that reflected the chosen art style while maintaining garment shape
- Interactive Flask app allowed iterative refinement of segmentation masks via foreground and background stroke inputs
- Generated designs demonstrated clear differentiation between VGG-19 style transfer and GAN-synthesized outputs
- Project was presented to professors at the NUS School of Computing and Senior Directors at Hewlett Packard Enterprise (HPE)

---

## ⚠️ Limitations

- GAN output quality is limited by the 128×128 resolution; higher resolution training would require significantly more compute
- Neural Style Transfer processing time is slow per image and is not suitable for real-time generation without optimization
- GrabCut segmentation requires manual user input; fully automatic garment isolation was not implemented
- The model is trained on art images only; generalization to diverse pattern types and textures has not been validated

---

## 🔮 Future Work

- **Higher resolution GAN training** — scale up to 256×256 or 512×512 for more detailed and wearable designs
- **Faster style transfer** — explore fast neural style transfer (feed-forward networks) to reduce per-image generation time
- **Automated segmentation** — replace GrabCut with a semantic segmentation model (e.g. U-Net) to remove the need for manual input
- **Diverse training data** — incorporate more varied art styles and pattern types to expand the design space
- **End-to-end pipeline** — integrate segmentation, style transfer, and GAN synthesis into a single user-facing interface

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/fashion-design-with-neural-style-transfer-and-gans.git
cd fashion-design-with-neural-style-transfer-and-gans

# Install dependencies
pip install flask opencv-python numpy keras tensorflow pillow
```

1. Run `image_resizer.py` to preprocess your art images into `art_data.npy`
2. Open `implementation.ipynb` to train the GAN model on the preprocessed art data
3. Run `run_app.ipynb` or execute `python __init__.py` to launch the Flask app
4. Go to `http://127.0.0.1:5000/` in your browser to use the interactive GrabCut tool
5. Add your own garment images under `static/images/<project_name>/_original/`

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
