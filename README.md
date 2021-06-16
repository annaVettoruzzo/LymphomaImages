# LymphomaImages

Project for Human Data Analytics course.

Deep Learning is used to classify lymphoma images into three different types:
Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL) and Mantle Cell Lymphoma (MCL). 
In the proposed scenario, convolutional neural networks (CNN) and recurrent neural networks (RNN) are used for classification purposes.
Different color spaces and different network architectures have been taken into account to reach the highest accuracy on a test set of images. 
The results reveal that the cascade of CNN and RNN layers with RGB images is able to extract the relevant features from each image 
and perform the final classification with good fidelity, despite the increase in the computation cost and overfitting. 
This model outperforms the traditional Machine Learning approaches reaching an accuracy higher than 97%,
despite the higher computational burden.

Code files:
- demo.py is a short demo to load the dataset in the selected color space, normalize it, load a pre-trained model and evaluate its performance;
- train.py is the complete code to load and process the dataset, fit the model selected and evaluate its performance.
- function.py contains all the additional functions used in the main code.

Participants:
- Anna Vettoruzzo
- Giulia Rizzoli

