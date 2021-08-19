# Data Science on TBC Microstructure Images

The code is part of the Juelich project work and is a result of with my master thesis titled "Porosity Characterization of Thermal Barrier Coatings using Image Analysis and Machine Learning". Link to the complete report: https://drive.google.com/file/d/17wLw9fepLLRAki7bkoBkRFN9Oxu-MHSU/view?usp=sharing

About the folders:

1.	ML_with_database: The porosity features were extracted using the first version of porosity classification based on the algorithm provided in the thesis was saved as JSON file. This dataset is used for the ML part and therefore generator function is not used. Porosity feature classification is constrained by time consumption to be used in the train/test data generation on the go
2.	ML_with_gen: All the features that could be used with a test/train generator are coded here for ML
3.	Pore_classification: The constant need for developing a better pore feature segmentation into three classes (points, threads, and clusters) are conducted here. It is also possible to add an ML code with the help of labelled ground-truth (manually using software such as labelme) that can segment these features using a model. But the requirements on image pre-processing such ‘ground-truth’ based generation will need further testing.

Installation:

1.	The code requires the portable 'Tesseract-OCR' folder (please check their website) to be placed in the root location NEXT TO THE CODE 
2.	The 'Data' folder should be placed similarly following the directory tree structure (see report)
Al the files and folder (with this tree structure) required to execute the code are uploaded in the Jülich repository (using sciebo). Please obtain the necessary permission to download this directory from the collaborators at IEK1-FZJ (Dr Daniel E Mack). A test directory is also provided in the Jülich sciebo with much lesser size (250 mb) which follows a directory tree-structure that will work for each of the codes with their corresponding microstructure images.



