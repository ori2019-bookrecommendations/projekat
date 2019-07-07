# Book recommendations system
Projekat iz predmeta "Osnovi računarske inteligencije", SIIT, 2019. godina.

Issue postavljen na: https://github.com/ftn-ai-lab/ori-2019-siit/issues/11

## Članovi tima:
* Kristina Đereg, SW36/2016

* Nemanja Janković, SW52/2016

## Instalacija aplikacije
```
python setup.py install
```

Delovi aplikacije mogu koristiti procesnu moć grafičke kartice ako ona to podržava.

Za spisak podržanog hardvera pogledati: https://www.tensorflow.org/install/gpu#hardware_requirements

Ako želite koristiti grafičku karticu potrebno je:
 
1. Instalirati tensorflow-gpu modul komandom:  ```pip install tensorflow-gpu ```

2. Ispratiti instrukcije instalacije potrebnih CUDA komponenti na linku: https://www.tensorflow.org/install/gpu#software_requirements

U suprotnom, potrebno je instalirati tensorflow modul komandom:  ```pip install tensorflow ```

------

Zbog veličine snimljenih modela, hostovani su na linku: https://drive.google.com/open?id=1Jygct7wLONBq80eVl-hu4OvxaJn-UVgH

Potrebno ih je smestiti u folder ```BookRecommendation/inputs/cf_models```


## Pokretanje aplikacije

* Pokretanje web servera za preporuke knjiga
```
python web.py
```

* Pokretanje konzolne preporuke knjiga
```
python predict.py -userID $userID -books $book1, $book2
```

* Pokretanje treniranja modela
```
python train.py cbf
python train.py svd
python train.py neumf $latent_factor_user $latent_factor_book $latent_factor_matrix_factorization $batch_size $number_of_epochs
```

* Pokretanje testiranja modela
```
python test.py
```
