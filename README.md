# Face Mask Detection

This project utilizes a deep learning model, based on the VGG16 architecture, to detect whether a person is wearing a mask or not. The model is trained on a labeled dataset with images of individuals wearing masks and without masks. The application uses OpenCV for real-time video capture and display, allowing it to detect and display mask status on a live camera feed.

## Features

- Real-time mask detection using webcam.
- Image pre-processing for standardized input dimensions.
- Pre-trained VGG16 architecture with a custom dense layer for binary classification (mask vs. no mask).
- User-friendly display of detection status with colored labels.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```
2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
   Make sure you have the following libraries:

     - Keras
     - OpenCV
     - NumPy
     - scikit-learn
  
 3. **Dataset Preparation:**

  - Create a dataset folder and place the training images in subdirectories:
       - `with_mask/` should contain images of people wearing masks.
       - `without_mask/` should contain images of people without masks.

  - Ensure that with_mask contains images of people with masks, and without_mask contains images without masks.

## Model Training
   The model uses transfer learning from the VGG16 network:

  1. VGG16 is loaded with pre-trained weights, and the final layer is removed.
  2. A new dense layer with sigmoid activation is added for binary classification.
  3. The model is then trained on the dataset images.
     
### Training Script

  ```python

   # Load and preprocess data
   # (Refer to the full code for details)

   # Train the model
   model.fit(X_train, Y_train, epochs=5, validation_data=(X_test, Y_test))
  ```
### Running Real-Time Detection
 1. Run the detection script:

  ```bash
     python detect_mask.py
  ```

 2. The model captures frames from the webcam, resizes them to the required input size, and predicts mask status. It displays:

   "Mask" with a green label if a mask is detected.
   "No Mask" with a red label if no mask is detected.
 3. Exit the webcam preview:
    Press x to close the webcam feed.

## Functions Overview
  ### detect_face_mask(img)
  - Purpose: Predicts mask status on an image.
  - Input: A pre-processed image.
  - Output: Mask status (1 for mask, 0 for no mask).
     
  ### draw_label(img, text, position, bg_color)
  - Purpose: Draws a label on the image.
  - Input:
       - img: The image frame.
       - text: The label text.
       - position: Coordinates for the label.
       - bg_color: Color of the label background.
       
   ### Main Loop
  - Continuously captures frames from the webcam.
  - Resizes each frame to the input shape (224x224) and predicts mask status.
  - Draws a label showing mask status with corresponding color.

## License
  This project is licensed under the MIT License.

## Acknowledgments
  - Keras for deep learning library.
  - OpenCV for real-time image processing.
  - VGG16 for pre-trained model architecture.
