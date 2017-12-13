## Veri madenciliği vize sonrası ders notları.

### 19 kasım (vize sonrası 1. hafta)
------------------------------------
**KNN (k neast neighbor)**
- Bilinen verilerin uzaklıklarına dayalı olarak yeni veri noktalarını sınıflandırmak için kullanılır.
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
- Boyutun kütüğü nedir? **// erim kütüğü ne demek ?**
    - Bir çok problemin boyut sayısının çok olduğu düşünülebilmektedir.
- Örneğin film tavsiye ederken her filmin kendi boyutuna sahiptir.Her bir film için derecelendirme vektörüdür.   
 ------------------------------------
 **K-means**
- Bu bir boyut azaltma örneğidir.
- Burada data k boyuta indirgenmiştir.
- Yüksek düzeyde matematik içerir.
    - **Principal Component Analysis**
        - Yüz tanıma , boyut indirme de kullanılır.
    - **4D ıris flowera verilerin görselleştirlmesi**
        - Scikit learn ile kullanılır.
        - Iris veri seti scikit_learn ile birlikte gelir.
        - İris çiçeği sepals ve petalslere sahiptir.
        - Birçok iris örnepği için sepals ve petalslerin uzunluğu ve genişliği bilinir.
        - PCA bunu 4 yerine 2 boyutta görselleştirmeye izin verir.

## Derste yazdığımız kod parçası
```bash
** dersteki örnek 1 **
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
        fields = line.rstrip('\n').split('|') #rstrip('\n') içindeki \n 'leri siler
        movie_id = int(fields[0])
        name = fields[1] #isimlerin olduğu sütünu alır.
        genres = fields[5:25] #5 den 25 sutuna kadar olan değerleri alır.
        genres = map(int,genres) #map(x,y) x'deki verileri y'deki verilere uygula
        movieDict[movie_id] = (name, np.array(list(genres)),
                 normRatings.loc[movie_id].get('size'),
                 grup.loc[movie_id].rating.get('mean'))
    
from scipy import spatial
import operator    

## İki film arasındaki uzaklığı bulan fonksiyon
 
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

** dersteki örnek 2 **

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

#print(pca. components)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

from pylab import *

columns = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure
```
#### devamını hoca gönderecek
#### //Ödev : Scikitlearn ile sınıflandırma örneği.
----------------------------------------------------------------------------------------------------------
### 26 kasım (vize sonrası 2. hafta)
------------------------------------
**Bu hafta ders olmadı.**
----------------------------------------------------------------------------------------------------------
### 3 aralık (vize sonrası 3. hafta)
------------------------------------
**Data Warehousing(Veri Ambarı)**
- Büyük veri merkezindeki veri tabanındaki bir çok kaynaktan veriler bulunur.
- "sql" ve "tsbleau" kullanılır sorgular için.(tableau verileri görsellleştirmek için kullanılıır.)
- Veri kaynaklarnı sürekli kılmak çok iş ve emek ister.
- Genellikle tüm depertmenlar da  biri veri ambarı bakımı yapmakla görevlidir.
- Büyük şirletler sık sık kullanır.
- Ölçeklendirme zor.
- Analiz edilecek verilerin aktarıldığı yerdir veri ambarı.
- Veri Normalizasyonu zordur
- Tüm veriler birbiri ile nasıl ilişkiye sahiptir? 
- İnsanlar nasıl görüşlere ihtiyaç duyuyor?
- Veri kaynaklarını sürekli kılmak çok iş ve emek gerektirir
-----------------------------------------------------------
**ETL Extract, Transform , Load(klasik yaklaşım)**
- ETL ve ELT verilerin bir veri ambarı içerisinden nasıl geldiğini ifade eder.
- ETL geleneksel(klasik yaklaşım) dir.
- Operasyonlar sistemlerde gelen ham verileri periyodik olarak çıkarmak.
- Veriler veri ambarı tarafından ihtiyaç duyulan şemaya dönüştürür.
- Son olarak dönüştürülen veriler veri ambarına yüklenir..
- Büyük verilerde dünüştürme işlemi büyük bir soruna dönüşebilir.
---------------------------------------------------------------
**ELT extract load transfırm(günümüzde çok fazla veri varsa kulanılandır.)**
- Big data için tek seçenek oracle değildir.
- Büyük verilerde işlem yaparken kırılganlığı(yablış yazmış olabilirim :D) önlemek için kullanılır.
- Hive gibi teknolojiler , bir hadoop kümesinde büyük veritabanına ev sahipliği yapmaya izin verieri.
- Büyük dağınık veriler nosql de saklanabilir. Spark veya mapreduce gibi teknolojilerle sorgulanabilirler.
- Hadoop un ölçeklenebilirliği, yükleme işleminde ona atmaya sağlar.
    - Önce ham veri extract edilir.
    - Sonra yüklenir.
    - Daha sonra hadoop gücü kullanılarak transform işlemi gerçekleştirilir.
- (casandra facebook un geliştiriği veri tabanıdır.Yatayda sınırsız veri araması ve veri çoksa performans artar.)
----------------------------------------------------------------
**Reinforcement Learninig** 
- Reinforcement Learning,Machine Learning'in ........ hesaba dayalı yaklaşımlar oluşturulmuştur.
- elifdemirtas.net/2016/08/20/reinforcementlearningnedir/ (hocanın kaynağı)
- Yukarıdaki adrese bak.
- Ödül , ceza işlemleriyle ilerlerme olur.
- Pacmen ve kedi-fare oyununda kullanılır. ilk öğrenir sonra uygular.
--Algoritmalar--
- Q-learninig
- Markov desicion process
- Dynamiv proggrqimng
----------------------------------------------------------------
**Bias/ varyans ikilemi(Bias / Variance Tradeoff)**
- Bir modelin genellleştirme hatası 3 farklı hatanın toplamı şeklinde ifade edilir.
1. Yanlılık(bias)
2. Varyans
3. İndirgenemez hata
-------------------
---------------------
Açıklamalar için (https://makineögrenimi.wordpress.com/2017/05/30/yanlilikvaryans-ikilemi-biasvariance-tradeoff/) bi bak istersen.
---------------------
**Bias** = bias modelinin ne kadar yanlş olduğunu ölçer. Örneğin veri ikinci dereceden bir polinom iken verinin lineer 
olduğunu varsaymak gibi. bias, modelin problemin çözümünü içermediğini gösterir.modelin zayof kaldığı bu durımda eksik öğrenme (under fitting) denir. yüksek biansa sahip modelin , eğitin verisini eksik öğrenme olasılığı fazladır.

**Varyans** = modelin tahmin ettiği verinin , gerçek verilein etrafında nasıl saçıldığını ölçer. varyns modelinin eğitim verisindeki düşük değişimlerdir.fazla veri varsa overfittingdir.

**İndirgenemez hata** = Verideki gürültülere bağlıdır.Bu hatayı azaltmanın tek yolu, veriyi temizlemektir.
- Bir modelin karmaşıklığını arttırmak , varyansını arttırır ve yanlılığını azaltır. Aksine , bir modelin karmaşılığını azaltmak, yanlılığı artırır ve varyansnı azaltır.(ikisininde normal olduğu en iyi durumdır.)
---------------------------------------------
**Kfold cross validation**
- Overfittingten kaçınmak için kullanılan bir yÖndemdir.
-------------------------------------------------
## Derste yazdığımız kod parçası
```bash
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets #iris  verisine erişmek için.
from sklearn import svm

iris = datasets.load_iris()

X_train , X_test, Y_train, Y_test = train_test_split(iris.data,
                                                     iris.target, test_size=0.4,
                                                     random_state=0)
# test_size = 0.4 elimizdeki verinin %40 ını test verisi için kullanacağımızı belittiik.
(Aşağı tarafı erimden aldım :D bende yoktu :) )

model = svm.SVC(kernel = 'linear', C=1).fit(X_train, y_train)
model.score(X_test, y_test)
sonuc = cross_val_score(model, iris.data, iris.target, cv=5)
print(sonuc)
print(sonuc.mean())

model = svm.SVC(kernel = 'poly', C=1).fit(X_train, y_train)
sonuc = cross_val_score(model, iris.data, iris.target, cv= 5)
print(sonuc)
print(sonuc.mean())
```
## ödev : Veri setini 5 e böl,her defasında birini eğitim,ötekileri veri seti olarak seçip sonuç bul bu sonuçların ortalamasını al
----------------------------------------------------------------------------------------------------------
### 10 aralık (vize sonrası 4. hafta)
------------------------------------
**Veri Önizlemesi**
**Kategorik Değişkenler**
-------------------------
- Bazı ML modelleri için kategorik değişkenlerin sayısala dönüştürülmesi gerekebilir.
---------------------
```bash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = pd.read_csv('Data.csv')

#girsi çıktı değişkenleri vardır.,
X = dataSet.iloc[:,:-1].values #son kalan haric hepsini al
y = dataSet.iloc[:, -1].values #son kalan değeri aldık

#missing data
#age ve salary alannlarında iki eksik bilgi var 
#-eksik olnalar silinebilir.
# -ortalama ya da medyanla doldurabiliriz.

#eksik veriler
from sklearn.preprocessing import Imputer

imputer =Imputer(missing_values='NaN',  #kayop verilerin nasıl oluştuğunu yazarız
                 strategy='mean', #most_frequery en çok tekrar eden alınır yazarsak strategy ye
                 axis=0) #eksik veri de 0 ises sütun , 1 ise satır buyunca bakarız.

imputer = imputer.fit(X[:, 1:3]) #3 sütünu almaz.
X[:, 1:3] = imputer.transform(X[:, 1:3])

#catagorik değişkenler
#bazı makine öğrenmeleri sade saısal basıları sözel verilerle çalışır.
#burada country ve purchased alanları mevcut.

#encode categorical data
from sklearn.preprocessing import LabelEncoder
labelEncode_X = LabelEncoder() #laberlerncoderden nesne oluşturduk.
X[:,0] = labelEncode_X.fit_transform(X[:,0]) # ülke isimlerine sağısal bir değer verdik..

#değerler rasgelr değerlerin bi önemi yok algoritmaya bunu anlatmamız lazım.

#dummy encoding
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()

#feature scalling
#ekimizdeki veriler belli bir aralığa getirmek.
#satandardisation , normalisation olamk üzer iki yöntem vardır. bunların formülleri vardı :D

#standartition
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#minmax 0 1 araında normaliza ememize yarar.
from sklearn.preprocessing import MinMaxScaler
mn_X = MinMaxScaler()
X = mn_X.fit_transform(X)

```
- Outliers 
- Veride bulunan aşırı değerlere verilen isimdir.
- Bunların veriden ayrılması gerekir.
- Bir kullanıcı çok film oylarsa bu herkesin oylarını etkileyebilir.
- Örneğin işbirlikçi filtrelemede binlerce filmi değerlendiren tek bir kullanıcı,herkesin oylarını etkileyebilir.Bu durum istenmez.Web log verilerinde botlar ve diğer ajanlar aykırı verileri temsil edebilir.

```bash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

gelirler = np.random.normal(27000,15000,10000) #¢fver,i oluşturduk.
gelirler = np.append(gelirler, [1000000000])
plt.hist(gelirler,50) # histogram grafiği çizdirdik.
plt.show() # ekranda gösterdik.

# -3 +3 standart sapma 
def  outlierCikar(data):
    o = np.median(data)
    s = np.std(data)
    filtered = [veri for veri in data
                if (o - 2 * s < veri < o + 2 * s)]
    return filtered

filtered = outlierCikar(gelirler)
plt.hist(filtered,50)
plt.show()
```
----------------------------------------------------------------------------------------------------------
