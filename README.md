## Veri madenciliği vize sonrası ders notları.

### 19 kasım (vize sonrası 1. hafta)
------------------------------------
**KNN (k neast neighbor)**
- Uzaklıklara dayalı olarak veri noktaları sınıflandırmaya yarar. 
- En çok kulllanılan öklik algoritmasıdır. manhatton diyede bir algoritma vardır.
- k değeri hangi sınıfa dahil olacağını belli eder.
- Supervised  learning  modeldir. 
------------------------------------
- rstrip('\n') : aradaki boşlukları temizler.
- map : şurdaki verileri şurdaki verilere uygula. gendes = map (int, gendes)  gibidir kullanımı.
------------------------------------
**Dimensionality Reduction**
- Boyutun önemi nedir ?
    - Bir çok problemin boyut sayısının çok oldğu düşünülebilmektedir.
 ------------------------------------
 **K-means**
- Bu bir boyut azaltma örneğidir.
- Burada data k boyuta indirgenmiştir.
 **Principal Component Analysis**
- Yüz tanıma , boyut indirme de kullanılır.
**4D ıris flowera verilerin görselleştirlmesi**
- Scikit learn ile kullanılır.

## Derste yazdığımız kod parçası
```bash
import pandas as pd #cvs dosyalarını okumak içn.
import numpy as np

u_columns= ['user_id','movie_id','rating'] # column lara isin verdik çünkü cvs dosyayısda isimli değil.

ratings = pd.read_csv(
        '/home/nafi/Desktop/3.sınıf_1.dönem/veri_madeniliği/7.hafta/u.data',
        sep='\t',names=u_columns, usecols=range(3)) #

grup = ratings.groupby('movie_id').agg({'rating':
    [np.size,np.mean]}) #kaç kişin oyladığını ve ortalama rating değerlerini aldık.
    
movieNumRatings = pd.DataFrame(grup['rating']['size']) #fimleri toplan kaç kişinin oyladığını görürüz.
normRatings = movieNumRatings.apply(lambda x :(x - np.min(x)) / (np.max(x) - np.min(x)))

movieDict={}

with open ('/home/nafi/Desktop/3.sınıf_1.dönem/veri_madeniliği/7.hafta/u.item',
           encoding= 'latin-1') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movie_id = int(fields[0])
        name = fields[1] #isimlerin olduğu sütünu alır.
        genres = fields[5:25] #5 den 25 sutuna kadar olan değerleri alır.
        genres = map(int,genres)
        movieDict[movie_id] = (name, np.array(list(genres)),
                 normRatings.loc[movie_id].get('size'),
                 grup.loc[movie_id].rating.get('mean'))
    
from scipy import spatial
import operator    

def computeDistance(a,b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA,genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

def getNeighbors(movieID, K):
    distances = [] # liste oluşturduk.
    for movie in movieDict:
        if(movie != movieID):
            dist = computeDistance(movieDict[movieID],movieDict[movie])
            distances.append((movie,dist)) #ilk parantez tuble diye ekleriz..
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 10 
avRating = 0
neighbors = getNeighbors(1, K)
for neighbor in neighbors:
    avRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor]))
    
avRating /=K


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle

iris = load_iris()

ornekSayisi , ozellikSayisi = iris.data.shape

print(list(iris.target_names))

X = iris.data
pca = PCA(n_components=2, whiten=True) fit(X) #2 componenete göre listeledik
X_pca = pca.transform(X)

print(pca.explained_variance_ratio_)
print(sum)
```



### 26 kasım (vize sonrası 2. hafta)
### 3 aralık (vize sonrası 3. hafta)
### 10 aralık (vize sonrası 4. hafta)
