<div align="center">
  <h1>The Dataset of RecAd</h1>
     
</div>


We provide two data set processing methods. Users can choose to directly download the data set we have processed from google drive and put it in the corresponding path location, or preprocess the raw data into the format RecAD need, and then use the corresponding function in recAD to process it.


## The datasets we provide in [GoogleDrive](https://drive.google.com/drive/folders/1MMvpFfbeSwt68DZdR7YP5BUJQzRc-LxW?usp=sharing)

* ml1m
* yelp
* Amazon-game
* Epinions
* Book-crossing
* BeerAdvocate
* dianping
* food
* ModCloth
* ratebeer
* RentTheRunway



## the Raw data format RecAD need

we recommend you use .csv file:
| user_id | item_id | rating | timestamp |
| --- | --- | --- | --- |
| shakja121 | 034545104X | 1 | 1455847245 |
| jkhsfd253 | 0155061224 | 0 | 1454771718 |

and then you can use the *raw_data* function in RecAD

```python
def raw_data(input_path, k_core, ratio, output_path, data_name):
    '''
    :param  input_path: The path of raw data in the format required by RecAD, input_path = '.\example.csv'
    :param      k_core: The minimum number of interactions between user and item, k_core = 10
    :param       ratio: Partition ratio of validation and test data set,  ratio=0.1
    :param output_path: The path to store the generated file, output_path = './book-crossing/'
    :param   data_name: The name of new dataset, data_name = 'ml1k'
    :return:Success or Reset the parameters
    '''
```



## The infomation about the Dataset we provided

|  **Dataset**  | **user_num** | **item_num** | **rating** | **k-core** |
|:-------------:|:------------:|:------------:|:----------:|:----------:|
|    Epinions   |     2465     |     2001     |    [1-5]   |      3     |
| Book-crossing |     3520     |     4414     |    [1-5]   |      6     |
|    dianping   |     33739    |     29665    |    [1-5]   |     10     |
|      Food     |     6707     |     11607    |    [1-5]   |     10     |
|    ratebeer   |     9113     |     34399    |    [1-5]   |     10     |
|      ml1m     |     5950     |     3702     |    [1-5]   |     10     |
|    ModCloth   |     9965     |      507     |    [1-5]   |     10     |
|  Beeradvocate |     10456    |     13845    |    [1-5]   |     10     |
|      yelp     |     54632    |     34474    |    [1-5]   |     10     |
|  Amazon-game  |     3179     |     5600     |    [1-5]   |     10     |
|   RentTheWay  |     4162     |     2890     |    [1-5]   |      5     |


## Acknowledgement

Thanks to the first two contributors: [@CS. Wang](https://github.com/Wcsa23187), [@Jianbai Ye](https://github.com/gusye1234)

