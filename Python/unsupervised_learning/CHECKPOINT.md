## Cel 
Analiza prównawcza jakości klasteryzacji różnych algorytmów przy użyciu różnych metryk z uwzględnieniem problemu doboru odpowiedniej liczby klastrów, braku skalowania danych oraz obserwacji odstających. Zastosowanie otrzymanych wyników do analizy rzeczywistych danych. 
## Jakich metryk użyjemy w projekcie do oceny modeli? 
Metryki użyte w naszym projekcie to:
- $SSE = \sum_{i=1}^{k} \sum_{x\in C_i} ||x-\mu_i||^2,$ gdzie $k$ oznacza liczbę skupień, $C_i$ zbiór punktów należących do $i$-tego skupienia dla $i=1,\ldots,k$ oraz, w przypadku metody k-means, $\mu_i$ to średnia arytmetyczna punktów w skupieniu $C_i$ (centroid) lub, w przypadku metody k-medoids, $\mu_i$ to odpowiednio zdefiniowana mediana punktów w skupieniu $C_i$ (medoid).
- *Calinski–Harabasz index* (https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index)
- *Silhouette measure* (https://en.wikipedia.org/wiki/Silhouette_(clustering))
- *Normalised Clustering Accuracy (NCA)*  (https://link.springer.com/content/pdf/10.1007/s00357-024-09482-2.pdf)
## Skąd pozysakmy dane?
Dane użyte w projekcie zostaną pozyskane z trzech źródeł. Użyte w projekcie syntetyczne dane będą pochodzić z dwóch źródeł. Pierwsze z nich to własnoręczne wygenerowane dane przy użyciu np. funkcji z pakietu *sklearn.datasets* takich jak *make_blobs*, *make_moons*. Kolejny to użycie danych benchamrkowych z pakietu **clustbench** (https://clustering-benchmarks.gagolewski.com/z_bibliography.html#id5). Zostaną one użyte do pierwszej części projektu, czyli analizy porównawczej algorytmów. Dane rzeczywiste zostaną pozysane ze stron https://www.openml.org/ oraz https://www.kaggle.com/datasets.
## Plan projektu
### Problem doboru odpowiedniej liczby klastrów
Problem doboru odpowiedniej liczby klastrów dotyczy algorytmów k-means, k-medoids oraz algorytmów hierarchicznych. W przypadku k-means oraz k-medoids przeanalizujemy ten problem przy użyciu metryki SSE, *Calinski–Harabasz index* oraz *Silhouette measure*. Spawdzimy jak zmieniają się one w zależności od liczby skupień. Dla algorytmów hierarchicznych wykorzystamy te same metryki, co poprzednio, a ponadto skorzystamy, z charakterystycznych dla tego algorytmu, dendrogramów. Dodatkowo dla zbiorów dwuwymiarowych oraz trójwymiarowych dokonamy oceny na podstawie wizualizacji zbioru.
### Problem danych nieprzeskalowanych oraz obserwacji odstających
Porównamy zachowanie algorytmów dla danych nieprzeskalowanych oraz tych po użyciu *StandardScaler*. Sprawdzimy jak zmieniają się w takiej sytuacji ww. metryki oraz czas wykonania algorytmu. Podobnej analizy dokonamy dla obserwacji odstających porórwnując metryki klasteryzacji zbiorów przed jak i po usunięcu obserwacji odstających. Dla danych dwuwymiarowych oraz trójwymiarowych dokonamy także oceny wizualnej zaproponowanej klasteryzacji.
### Analiza porównawcza algorytmów 
Analiza porównawcza algorytmów zostanie wykonana dla zróżnicowanych syntetycznych zbiorach danych różniących się wymiarem, liczbą rekordów oraz strukturą. Użyte zostaną tutaj metryki *Calinski–Harabasz index*,
*Silhouette measure*, *Normalised Clustering Accuracy*, które pozwolą nam ocenić, który z algorytmów uzyskał najlepsze wyniki. Ponadto porównamy czas wykonania algorytmów.
### Wykorzystanie w eksploracji danych rzeczywistych 
Sprawdzimy zdolność do tworzenia nowych klas podziału oraz ich interpretowalność na rzeczywistych zbiorach danych. Ponadto przetestowana zostanie możliwość użycia wyników segmentacji jako nowych zmiennych objaśniających oraz podejście w postaci oddzielnej budowy modeli predykcyjnych dla każdego klastra. Poprawę zdolności predykcyjnych zweryfikujemy za pomocą metryki *balanced accuracy* lub *MSE* na zbiorze testowym. 
## Organizacja pracy 
### Podział
* Mateusz Brochocki - znajdywanie optymalnej liczby klastrów, eksperymenty na danych rzeczywistych  
* Michał Dębski -  znalezienie oraz wygenerowanie zbiorów danych oraz analiza porównawcza algorytmów klasteryzacji  
* Aleksander Karch - implementacja metryk i narzędzi potrzebnych do analizy algorytmów grupowania danych  
* Adrian Krzyżanowski - wpływ skalowania oraz obserwacji odstających na działanie algorytmów 

### Kamienie milowe
* 04.05 - przygotowanie wszystkich potrzebnych zbiorów danych i dokonanie wstępnych analiz 
* 24.05 - wykonanie wszystkich analiz i wizualizacji 
* 07.06 - gotowy raport i prezentacja
