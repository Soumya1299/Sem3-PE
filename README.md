**Acknowledgment
This work was done as a part of Project Elective with Dr. Kurian Polachan, IIIT-Bangalore.
**

To: Run inference of a IRIS dataset in PSoC-6 using Modustoolbox ML APIs(The inputs are hardcoded)

Work Done: Multiclass classification model is built on IRIS dataset to classify among the 3 classes of IRIS flower namely setosa, versicolor & virginica. 
Tools used: Google Collab, ModudToolbox, PuTTy
Overview:
•	Built the model.
•	Trained the model.
•	Converted the model into tflite format.
•	Converted the Ml model into flatbuffer format, .h file
•	Wrote the firmware in modustoolbox to run the ML model in PsoC. Used ML libraries provided by Modustoolbox & added path of models in makefile. Did some more changes in make file like adding C++ flags and other software ML components.
•	Built the main.cpp and run the model on microcontroller.
•	Used Putty terminal to display the output.

Details
1.Built the deep neural network model of IRIS multiclassification in google collab:
•	loaded the Iris dataset from python’s tensorflow(150 samples, 3 classes).
•	Features are 4-dimensional floats, labels are integers.
•	Dataset is split into: Training (60%), Validation (20%), Test (20%)
•	StandardScaler is used which normalizes each feature (mean=0, std=1).
•	One-hot Encode Labels : Converts integer labels (0/1/2) into one-hot vectors ([1,0,0], [0,1,0], [0,0,1]) which are needed for categorical crossentropy loss.
•	Built Neural Network with Input layer: 4 features. Hidden layers: 16 and 12 neurons (ReLU); Output: 3 neurons (softmax) and compiled with Adam optimizer and categorical crossentropy loss.

2.Trained the model in google collab
•	Trained the model for 100 epochs, batch size 5.
•	Used validation set for progress check.
•	Saved .keras model.
•	Model predicts probabilities: chooses the class with highest probability.
•	Then calculate: Accuracy, Precision / Recall / F1 (macro average), Classification Report & Confusion Matrix which show how well the model performs per-class.
3.Converted the full Keras model into lightweight TensorFlow Lite model (.tflite).
4.Converted the model into flat buffer format and saved it as header file. This lets us embed the model directly into microcontroller firmware.
 
5. In ModusToolbox, created a new application in which I wrote the source code as a .cpp file. Included .tflite & .h files for the DNN model. The firmware code has hardcoded values of petal length, petal width, sepal length, sepal width mentioned in the inference code. The function takes the hardcoded values and predicts the output based on the trained models incorporated.
Used libraries: ML-Inference, ML-Middleware, ML-TFLite micro to integrate Tensorflowlite micro.
 Did some changes in the makefile.
 
6. Built the project.
   
Changes Done in Makefile: 
1.TARGET defined:
•	Defined the Board Support Package (BSP): CY8CKIT-063-BLE.
2. APPNAME: 
•	Name of the app(name of final binary)
3. COMPONENTS
•	Added software components: ML_TFLM: TensorFlow Lite for Microcontrollers & ML_FLOAT32 is used floating-point support for inference 
4. SOURCES
•	Additional source files to compile.
5. INCLUDES
•	Adds ./models to the include path.
6. DEFINES
•	Adds preprocessor defines.
•	Here: TF_LITE_STATIC_MEMORY uses static memory allocation for TFLite Micro.

 
Inference of the tinyML model is run on CY8CPROTO-063BLE board. COM-4 port and putty terminal is used to display the output communicated from PsoC to PC via UART.

