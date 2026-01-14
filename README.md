# Progetto di Machine Learning

<p style="text-align: center;">
<b>Sabato Iaquino (Matricola 0512123029) Anno Accademico 2025-2026
</br>Corso di Machine Learning tenuto dai professori Giuseppe Polese e Loredana Caruccio
</p>

## Specifiche per lo svolgimento del progetto

Il progetto ha l’obiettivo di realizzare una (piccola) applicazione/soluzione di Machine Learning (ML), la quale potrà essere svolta in gruppo (di massimo due persone) ed esposta ai docenti del corso.\
La soluzione proposta dovrà applicare, nei vari step della pipeline di ML, gli approcci e le tecniche studiate nell’ambito del corso. Ad esempio, verranno valutate positivamente una corretta analisi e risoluzione delle problematiche legate alla presenza di missing value, di outlier, di dati sbilanciati, feature ridondanti, ecc.\
Inoltre, risulta importante giustificare la selezione del/dei modello/i, valutare le performance (almeno con le metriche di valutazione tradizionali) e i tipi di errore effettuati dal modello.\
Nota: non è strettamente necessario realizzare soluzioni che impieghino i soli modelli studiati nell’ambito del corso; tuttavia, è necessario
dimostrare padronanza nell’esposizione di tutte le tecniche impiegate per lo sviluppo del progetto.\
Il progetto dovrà essere consegnato tramite il link messo a disposizione sulla piattaforma e-learning allegando un file .zip che contenga il codice (o eventualmente un link ad un repository GitHub accessibile) e un report di progetto nel quale venga dettagliato:

1. Lo scenario/problema/task analizzato.
2. Il dataset utilizzato e le sue caratteristiche.
3. Le issue analizzate, le soluzioni proposte e le scelte di progettazione effettuate.
4. L’analisi delle prestazioni, che comprenda dati numerici e grafici.
5. Considerazioni finali e possibili sviluppi futuri.

_N.B.: I risultati del progetto dovranno essere presentati nelle sessioni di discussione attraverso una presentazione (Power Point, Google Slides, ecc.)._

## Dataset
Il dataset utilizzato è Credit Card Fraud Detection di Machine Learning Group of Université Libre de Bruxelles, disponibile su Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Componenti del progetto
All'interno della cartella di progetto sono disponibili:

- `data_analysis.py` contiene l'analisi del dataset;
- `analysis_output` cartella generata da `data_analysis.py` contenente il risultato dell'analisi del dataset;
- `preprocessing.py` contiene le operazioni di pre-processing e il feature engineering;
- `preprocessing_output` cartella generata da `data_analysis.py` contenente il risultato delle operazioni di pre-processing e di feature engineering;
- `model_training.py` contiene le operazioni di addestramento dei modelli Gaussian Naïve Bayes e Random Forest, con l'analisi e il paragone delle performance;
- `models_output` cartella generata da `model_training.py` contenente il risultato delle operazioni di training.

## Documentazioni disponibili
All'interno della cartella di progetto sono disponibili:

- `Documentazione.pages` documentazione testuale del progetto in formato Apple Pages;
- `Documentazione.pdf` documentazione testuale del progetto in formato Portable Document Format;
- `Presentazione.key` presentazione del progetto in formato Apple Keynote;
- `Presentazione.pdf` presentazione del progetto in formato Portable Document Format.