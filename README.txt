Before executing, get the ETL-1 dataset from http://etlcdb.db.aist.go.jp/, by registering before downloading.
Also, you have to download the sample script "unpack.py" from the same website.



Running instructions:
1. execute unpack.py to generate .png images from the raw data files
	(you might have to set the additional modifier 'ignore' in get_char() (line 68) for the program to run normally.)
	The program is run by the command 'python unpack.py ETL1C_XX', with XX being the file you want to unpack.
2. execute extract.py to read the images into a usable format and save as data.pt
3. execute preprocess.py. This reads data.py and produces 6 files with the preprocessed train, dev, and test set (x and y)
4. model_util.py contains the model generation and does not need to be executed
5. execute train.py to train the network
6. execute evaluate.py for producing the results presented in the paper. This will print the accuracy and generate several plots


