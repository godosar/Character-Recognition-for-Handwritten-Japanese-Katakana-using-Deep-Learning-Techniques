import numpy as np
import imageio
import torch



x_res = int(3200/50) #pixel width of each entry =64
y_res = int(2520/40) #pixel height of each entry =63
res = 64 #consider a square image
n_katakana = 51 #number of characters in the dataset
n_writers = 1411 #number of entries per character



def split_image(im):
    '''
    Reads a full image array and splits it into seperate characters.
    Each of the png files contain 40*50=2000 entries.
    '''
    images = np.zeros((2000,res,res))
    for i in range(40): #index for rows
        for j in range(50): #index for columns
            images[i*50+j] = np.concatenate([im[i*y_res:(i+1)*y_res, j*x_res:(j+1)*x_res], np.zeros((1,64))]) #add an empty row to get the resolution to be square
    return images



def read_data():
    '''
    reads the full dataset of all 51 katakana from the .png files produced by unpack.py and puts them into an array.
    Two individual characters are duplicated (they were missing in the dataset).
    '''
    data = np.zeros((n_katakana,n_writers,res,res))
    print('extracting ETL1 dataset')

    for i in range(7,14): #7-13 are the files with katakana
        
        n_files, n_char = 6, 8 #number of pngs produced from one file, number of characters contained in those pngs
        #consider the different length of the last file
        if i == 13: n_files, n_char = 3, 3

        #iterate over each image produced from one file ('chunk'), each chunk will contain n_char characters from n_files pngs
        chunk = None
        for j in range(n_files):

            #read all the png files
            if i in range(7,10):
                im = imageio.imread(f'ETL1/ETL1C_0{i}_0{j}.png')
            elif i in range(10,14):
                im = imageio.imread(f'ETL1/ETL1C_{i}_0{j}.png')
            im = split_image(im)

            #concatenate every array from a contained chunk
            if j == 0:
                chunk = im
            else:
                chunk = np.concatenate([chunk,im])
            print(f'>>>{j}/{n_files}',end='\r')

        #for the 2 missing characters, insert a copy of another writer to replace it
        if i == 9: #one character 'na' is missung (5th character)
            #shift all following values one to the right
            chunk[5*n_writers:8*n_writers] = chunk[5*n_writers-1:8*n_writers-1]
            #insert another appearance of the character, here: the previous one
            chunk[5*n_writers-1] = chunk[5*n_writers-2]
        if i == 12: #one character 'ri' is missung (2nd character)
            chunk[2*n_writers:8*n_writers] = chunk[2*n_writers-1:8*n_writers-1]
            chunk[2*n_writers-1] = chunk[2*n_writers-2]
        print(f'file {i-6}/7')

        #insert each chunk into the data array
        for k in range(n_char):
            data[(i-7)*8+k] = chunk[k*n_writers:(k+1)*n_writers]

    data = torch.from_numpy(data.astype(np.float32))
    print('done!')
    return data

if __name__ == '__main__':
    d = read_data()
    torch.save(d,'data/data.pt')
