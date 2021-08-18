# Data Science on TBC

The code is part of the Juelich project work and is a result of with my master thesis titled "Porosity Characterization of Thermal Barrier Coatings using Image Analysis and Machine Learning"

Link to the complete report: https://drive.google.com/file/d/17wLw9fepLLRAki7bkoBkRFN9Oxu-MHSU/view?usp=sharing

1) ML_with_database: The porosity features were extracted using the first version of porosity classification based on the algorithm provided in the thesis was saved as JSON file. This dataset is used for the ML part and therefore generator function is not used. Porosity feature classification is constrained by time consumption to be used in the train/test data generation on the go

2) ML_with_gen: All the features that could be used with a test/train generator are coded here for ML

3) Pore_classification: The constant need for developing a better pore feature segmentation into 3 classes (points, threads and clusters) are conducted here. It is also possible to add an ML code with the help of labelled ground-truth (manually using software such as labelme) that can segment these features using a model. But the important requirements with respect to image pre-processing for such ground-truth generation needs to be further improved and tested.

The code requires the 1) 'Tesseract-OCR' folder (please check their website) to be placed in the root location NEXT TO THE CODE and 2) the 'Data' folder placed similarly
Al the files and folder struture required to execute the code are uploaded in the Juelich repository (using sciebo). Please obtain the necessary permission to download this directory from the concerned collaborator. A test directory is also provided in the juelich sciebo with much lesser size (250 mb) which has a datastructure that could work for each code with example images.


