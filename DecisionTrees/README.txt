'''
Created on Sep 20, 2015

@author   : NiharKhetan, Ghanshyam Malu
@desc     : Decision Tree Modeler
            Train the model, Test it and plot the confusion matrix
            Also, visualizes it using the matplotlib.pyplot 
@Usage    : Execute the python file to run the Decision Tree Modeler
			$ python Run.py

@Version  : Uses Python 2.7

@Sample Output

==========================================================================================
			Welcome to Decision Tree Modeler
==========================================================================================

Choose one of the available options:

	0 - Model, Train and Test 'Zoo' data
	1 - Model, Train and Test 'Carvana' data (personal dataset) 

Your choice...(0 or 1):  0

Change default settings for Max Tree Depth, Display log (Y/N)?... n



******************************************************************************************
				 Decision Tree
******************************************************************************************

------------------------------------------------------------------------------------------
Attribute: root 	Feature: f11 	Level: 0 	Children Count: 5
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Feature	: f10              	Level: 1
	Attribute: 2                    	Feature	: f0               	Level: 1
	Attribute: 4                    	Feature	: f2               	Level: 1
	Attribute: 6                    	Feature	: f4               	Level: 1
	Attribute: 8                    	Label	: 7                	Level: 1 	 ClassificationLabel: 7 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 0 	Feature: f10 	Level: 1 	Parent Feature: f11 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Feature	: f6               	Level: 2
	Attribute: 1                    	Feature	: f1               	Level: 2

------------------------------------------------------------------------------------------
Attribute: 0 	Feature: f6 	Level: 2 	Parent Feature: f10 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Label	: 7                	Level: 3 	 ClassificationLabel: 7 	****** Label *****
	Attribute: 1                    	Label	: 3                	Level: 3 	 ClassificationLabel: 3 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 1 	Feature: f1 	Level: 2 	Parent Feature: f10 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Label	: 1                	Level: 3 	 ClassificationLabel: 1 	****** Label *****
	Attribute: 1                    	Label	: 4                	Level: 3 	 ClassificationLabel: 4 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 2 	Feature: f0 	Level: 1 	Parent Feature: f11 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Label	: 1                	Level: 2 	 ClassificationLabel: 1 	****** Label *****
	Attribute: 1                    	Label	: 2                	Level: 2 	 ClassificationLabel: 2 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 4 	Feature: f2 	Level: 1 	Parent Feature: f11 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Feature	: f6               	Level: 2
	Attribute: 1                    	Label	: 1                	Level: 2 	 ClassificationLabel: 1 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 0 	Feature: f6 	Level: 2 	Parent Feature: f2 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Label	: 7                	Level: 3 	 ClassificationLabel: 7 	****** Label *****
	Attribute: 1                    	Label	: 5                	Level: 3 	 ClassificationLabel: 5 	****** Label *****

------------------------------------------------------------------------------------------
Attribute: 6 	Feature: f4 	Level: 1 	Parent Feature: f11 	Children Count: 2
------------------------------------------------------------------------------------------ 

	***Children***
	Attribute: 0                    	Label	: 6                	Level: 2 	 ClassificationLabel: 6 	****** Label *****
	Attribute: 1                    	Label	: 7                	Level: 2 	 ClassificationLabel: 7 	****** Label *****

==========================================================================================
					ERROR RATE
==========================================================================================

	Incorrect Classification Count: 3 	Correct Classification Count: 32

	*************************************************************************
				Error Rate is 8.571 %
	*************************************************************************




==========================================================================================
					ACCURACY
==========================================================================================

	Incorrect Classification Count: 3 	Correct Classification Count: 32

	*************************************************************************
				Accuracy is 91.429 %
	*************************************************************************




==========================================================================================
				CONFUSION MATRIX
==========================================================================================


****************************** Predicted Values As Columns ******************************
****************************** Expected Values As Rows     ******************************

		: 	1 	| 	3 	| 	2 	| 	5 	| 	4 	| 	7 	| 	6 	| 
---------------- ----------------------------------------------------------------------------------------------------------------
	1 	: 	14 	| 	0 	| 	0 	| 	0 	| 	0 	| 	0 	| 	0 	| 
	3 	: 	0 	| 	0 	| 	0 	| 	1 	| 	0 	| 	1 	| 	0 	| 
	2 	: 	0 	| 	0 	| 	7 	| 	0 	| 	0 	| 	0 	| 	0 	| 
	5 	: 	0 	| 	0 	| 	0 	| 	1 	| 	0 	| 	0 	| 	0 	| 
	4 	: 	0 	| 	0 	| 	0 	| 	0 	| 	4 	| 	0 	| 	0 	| 
	7 	: 	1 	| 	0 	| 	0 	| 	0 	| 	0 	| 	3 	| 	0 	| 
	6 	: 	0 	| 	0 	| 	0 	| 	0 	| 	0 	| 	0 	| 	3 	| 


------------------------------------------------------------------------------------------
		Thank you for using the Decision Tree Modeler !
------------------------------------------------------------------------------------------ 



'''	