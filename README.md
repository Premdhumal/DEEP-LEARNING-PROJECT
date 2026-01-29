# DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: Prem Dilip DhumaL

INTERN ID: CTIS1752

DOMAIN: Data Science

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

#DESCRIPTION:
In this task, I implemented a deep learning model for image classification using PyTorch. The main objective of this task was to understand how deep learning models work and how they can be used to classify images. Image classification is one of the most common applications of deep learning and is widely used in many real-world systems such as face recognition, handwritten digit recognition, medical image analysis, and self-driving cars.

For this project, I used Visual Studio Code (VS Code) as my development environment. VS Code helped me write Python code, manage project files, and run the program using the integrated terminal. It also made debugging and organizing the project easier.

The programming language used for this task was Python, because it is simple, easy to understand, and highly supported for machine learning and deep learning applications. The main library used was PyTorch, which is a popular deep learning framework widely used in research and industry. Along with PyTorch, I used Torchvision for loading image datasets and Matplotlib for visualizing results. NumPy was also used for handling numerical data and arrays.

For the dataset, the MNIST handwritten digit dataset was used. This dataset contains thousands of grayscale images of digits from 0 to 9. Each image is of size 28×28 pixels. The dataset was automatically downloaded using Torchvision, which made data loading simple and efficient.

The first step of the task was data preprocessing. The images were converted into tensors and normalized so that the pixel values remain within a standard range. This step is important because normalized data helps the model train faster and produce better results.

After preprocessing, a deep learning neural network model was created. The model consisted of multiple fully connected layers. The input layer received the image data, which was flattened into a one-dimensional vector. Hidden layers with ReLU activation functions were used to learn patterns from the images. The final output layer contained ten neurons, representing digits from 0 to 9.

The model was trained using the Adam optimizer, which adjusts the learning rate automatically during training. The loss function used was Cross Entropy Loss, which is commonly used for multi-class classification problems. The training process was performed for several epochs, and during each epoch, the model learned from the training images and updated its parameters.

During training, the loss value was recorded for each epoch. These values were then visualized using a line graph to show how the loss decreased as training progressed. This visualization helped in understanding whether the model was learning properly.

After training, the model was tested on unseen data. Sample images from the test dataset were passed through the trained model, and predicted labels were displayed along with the images. This helped verify the accuracy and performance of the model visually.

This project demonstrates the basic workflow of a deep learning image classification system. Such models can be applied in real-world applications like document scanning, optical character recognition, image-based authentication systems, and intelligent monitoring systems.

Through this task, I gained practical understanding of deep learning concepts such as neural networks, activation functions, loss functions, optimizers, and training loops. This task improved my confidence in working with PyTorch and provided hands-on experience in building and visualizing deep learning models.
