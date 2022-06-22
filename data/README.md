# Elenco dei files
- noleggibit_id.parquet
- noleggibit_storico.parquet
- noleggi_idstatus.parquet
- noleggi_status.parquet
- noleggitier_idtrips.parquet
- noleggitier_trips.parquet
- noleggiwinds_idstatus.parquet
- noleggiwind_trips.parquet
- stato_mezzi_gbfs_now.parquet
- stato_mezzi_gbfs.parquet
- stato_mezzi_now.parquet
- stato_mezzi.parquet
- stato_mezzi_wind_now.parquet
- stato_mezzi_wind.parquet
- stato_mezzi_windpul.parquet
- tripbit_deda.parquet
- trip_dedav3.parquet
- trip_error.parquet
- tripsbit_deda.parquet
- tripsbit_point_error.parquet
- tripsbit_point.parquet
- tripsbit_v3.parquet
- trips_deda.parquet
- trips_dedav3.parquet
- trips_error.parquet
- trips.parquet
- trips_point_error.parquet
- trips_point.parquet
- trips_pointv3.parquet
- trips_v3.parquet
- tripsv3.parquet

## Prefisso dei nomi dei file
Alcune keywords importanti:
- bit, wind, tier: sono nomi di operatori
- deda: dati rielaborati dall'azienda DedaGroup
- gbfs: segue lo schema gbfs
- error: contiene errori?
- point: contiene coordinate?

### Noleggi
Informazioni sul noleggio:
- `noleggibit_id.parquet`: contiente informazioni sul noleggio di e-scooter dell'operatore BIT. I campi descritti sono _lastid_, _maxid_, _incid_ e _stato_. _Stato_ è una variabile binaria che assume valori 0 o 1.
- `noleggibit_storico.parquet`: contiene lo storico dei noleggi dell'operatore BIT, dal 2020-11-30 al 2022-02-19. Contiene i campi: 
    - _id_ (numpy integer, identificativo della corsa);
    - *start_time* e *stop_time* (di tipo str; contenenti i giorno e orario di inizio e fine della corsa);
    - *inizio* e *fine* (di tipo pd.Timestamp; contententi i timestamp di inizio e fine corsa);
    - *start_tmda* e *stop_tmal* (di tipo datetime.time; contenti orario in ore-minuti-secondi rispettivamente di inizio e fine corsa);
    - *giorno_set* (numpy integer; indica il giorno della settimana in cui la corsa è stata svolta);
    - *break_time* (numpy integer; indica il tempo in secondi della sosta??);
    - *start_latitude*, *start_longitude*, *stop_longitude* e *stop_latitude* (type decimal.Decimal; indicano rispettivamente la latitudine e la longitudine a inizio corsa e a fine corsa);
    - *trip_km* (decimal.Decimal; indica i chilometri percorsi durante il viaggio);
    - *data_start* e *data_stop* (type datetime.date;indicano rispettivamente la data, in formato anno-mese-giorno, di inizio e fine corsa);
    - *tempo* (None values);
    - *ora_start* e *ora_stop* (type str; contententi ora-minuti-secondi di inizio e fine corsa);
    - *sec_start* e *sec_stop* (numpy integer; orario di inizio e fine corsa in secondi);
    - *sec_diff* (numpy integer; differenza tra *sec_start* e *sec_stop*), *sec_diffeff* (?);
    - *min_diff* (integer; differenza tra *sec_start* e *sec_stop* in minuti);
    - *min_secdiff* (?);
    - *device_id* (integer; unique identifier del dispositivo), *fascia_standard* (integer compreso tra 0 e 5);
    - *fascia_stdes* (str; assume valori '12-16', '8-12', '20-22', '16-20', '6-8', 'Fuori fascia');
    - *fascia_custom* (integer compreso tra 0 e 7);
    - *fascia_csdes* (str; assume valori '12-14', '9-12', '7-9', '19-22', '17-19', '14-17', 'Fuori fascia', '6-7');
    - *fascia_standard_it1* (integer compreso tra 0 e 5);
    - *fascia_stdes_it1* (str; assume valori '12-16', '8-12', '20-22', '16-20', 'Fuori fascia', '6-8');
    - *fascia_custom_it1* (integer compreso tra 0 e 7);
    - *fascia_csdes_it1* (str; assume valoro '14-17', '12-14', '9-12', '19-22', '17-19', '7-9', 'Fuori fascia', '6-7');
    - *fascia_standard_it2* (integer compreso tra 0 e 5);
    - *fascia_stdes_it2* (str; contiene valori '12-16', '8-12', '20-22', '16-20', '6-8', 'Fuori fascia');
    - *fascia_custom_it2* (integer compreso tra 0 e 7);
    - *fascia_csdes_it2* (str; assume valori '14-17', '12-14', '9-12', 'Fuori fascia', '19-22', '17-19', '7-9', '6-7');
    - *tempoda* e *tempoal* (type decimal.Decimal; tempo da/a ??);
    - *tempodiff* (None values);
    - *device_name* (stringa alfanumerica indicante il nome del dispositivo).
- `noleggi_idstatus.parquet`: stato dei noleggi. Contiene: _operator_ (str; BIT o VENTO), _prog_ e _endtime_ (entrambi contententi NaN values), _lastdt_ (None values), *var_data* (type string; data, nel formato anno-mese-giorno), *var_tempo* (string; 'T'+ ora del giorno, da '00' a '23'), *stato* (int; assume valori 0 o 1 e indica lo status del noleggio al tempo e giorno indicato nei rispettivi campi)
- `noleggi_status.parquet`: 

### Stato_mezzi
stato dei mezzi

### Trip
percorso

### Trips
Dati dei percorsi:

# trips_pointv3.parquet
Contiene i seguenti campi:
- `operator` (str): operatore che fornisce il servizio. Assume valori 'BIT' o 'VENTO'.
- `point_operator_id` (str): 'Bit Mobility' o 'Tier'.
- `point_trip_id` (str): identificativo della corsa (alcuni identificativi si ripetono a distanza di tempo).
- `point_timestamp` (datetime): indica giorno e orario in cui è stata rilevata l'attività.
- `point_latitude` e `point_longitude`: coordinate in cui si trovava il mezzo al momento riportato nel corrispettivo timestamp.
- `point_sequence` (int): ordine dei punti visitati all'interno di una stessa corsa.
- `point_reliability`: None values.

# trips_pointv3_cleaned.parquet
Contiene il risultato delle operazioni di cleaning effettuate sul dataset `trips_pointv3.parquet` attraverso il notebook `0_clean_data_trips_pointv3.ipynb`.

# tripsv3.parquet
Contiene i seguenti campi:
- `operator`: indica l'operatore che fornisce il servizio (BIT o VENTO).            
- `trip_operator_id`: identificativo dell'operatore: assume valori "Bit Mobility", "WIND" e "Tier", dove Wind e Tier rappresentano lo stesso operatore, in quanto Wind è stata acquisita da Tier che a sua volta è stata acquisita da Vento.      
- `trip_vehicle_id`: identificativo univoco del mezzo (e-scooter); sono presenti 645 e-scooters.             
- `trip_id`: identificativo della corsa; persiste la problematica della ripetizione degli ID dopo qualche tempo.    
- `trip_start` e `trip_end`: Timestamp di inizio e fine della corsa.        
- `trip_start_epoch` e `trip_end_epoch`: Decimal type; identificano tempo di inizio e fine della corsa (ridondante).              
- `trip_origin_time` e `trip_destination_time`: datetime type; informazione ridondante circa il tempo di inizio e di fine corsa.
- `trip_origin_latitude` e `trip_origin_longitude`: Decimal type rappresentante le coordinate del mezzo al momento della partenza.
- `trip_destination_latitude` e `trip_destination_longitude`: Decimal type rappresentante le coordinate del mezzo al momento dell'arrivo.   
- `trip_points_num`: Decimal type; numero di punti registrati durante la corsa.            
- `trip_points_numall`: Decimal type; per lo più riporta la stessa informazione di "trip_points_num".   
- `trip_length`: float riportante la distanza percorsa in metri.    
- `trip_mode`: str; indica il tipo di mezzo utilizzato. Assume unicamente valore "scooter".             
- `trip_accuracy`: float, assume valori 15., 1. e nan.
- `trip_duration_break_excluded`: Decimal type.
- `trip_user_id`: all None values.
- `trip_properties`: all None values. 

# trips_pointv3_cleaned.parquet
Contiene il risultato delle operazioni di cleaning effettuate sul dataset `tripsv3.parquet` attraverso il notebook `0_clean_data_tripsv3.ipynb`.


# Additional data

# google_transit_urbano.zip
Contiene i files relativi al trasporto pubblico della provincia di Trento in formato GTFS, ricavati dal sito di [TrentinoTrasporti](https://www.trentinotrasporti.it/open-data).

# all_bus_alternatives.parquet
File risultante dall'analisi di possibili alternative ai viaggi in e-scooter, effettuate all'interno del notebook `cfr_autobus.ipynb`.
Riporta:
- `escooter_id`: identificativo univoco della corsa in e-scooter 
- `bus_trip_id`: identificativo del possibile trasporto alternativo
- `bus_stops`: coordinate dei punti appartenenti al percorso svolto dal mezzo di trasporto alternativo

# map_matched_routes.parquet
Risultato del map-matching delle tratte in e-scooter effettuato con Valhalla Docker all'interno di `MapMatching_ValhallaDocker.ipynb`.
Contiene:
- `unique_id`: identificativo univoco della corsa in e-scooter
- `route`: coordinate, a seguito del map-matching, del percorso svolto in e-scooter 
- `matched`: variabile buleana, True se il map-matching ha avuto successo, altrimenti False. In caso assuma valore False, le coordinate riportate in `route` sono quelle originarie, precendenti al map-matching.

# ways_id.json
File json prodotto in `interactive_tratte_forti.ipynb` per effettuare il match tra l'identificativo del segmento stradale restituito da Valhalla Docker e le coordinate che lo caratterizzano. Associato ad ogni identificato, presenta una lista di coordinate e il conteggio delle volte in cui quella strada è stata percorsa, secondo le osservazioni effettuate sui monopattini elettrici nel periodo di raccolta dati. UPDATED: per ogni identificativo è stata aggiunto ai valori associati il nome della strada che rappresenta.

# map_matched_edges_ids.parquet
Prodotto all'interno di `interactive_tratte_forti.ipynb`. Contiene: `unique_id`, `route`, `matched` (boolean), `dates` (dt.date), `month` (int), `year` (int), `edge_ids` (lista dei segmenti stradali attraversati; ognuno appare un'unica volta, anche quando viene attraversato molteplici volte).

# mm_wayID_speed_name.parquet
Dataset con dati delle route successivi al map-matching, sottoposti nuovamente a Valhalla Docker per ottenere gli identificativi dei segmenti stradali e le velocità per corsa per segmento, oltre che ai nomi delle strade o piazze corrispondenti al segmento. Oltre alle velocità ricavate dalla request a Valhalla, è stato utilizzato pyproj.Geod per trovare la distanza percorsa, e successivamente da questa calcolare la velocità complessiva dell'intera corsa.
Le coordinate relative ai segmenti potranno poi essere ricavate consultando ways_id.json.
Podotto da *produce_dataset_speed_by_vayID.py*. 
Le variabili contenute nel dataset sono:
- `unique_id`: identificativo univoco della corsa in e-scooter
- `route`: coordinate, a seguito del map-matching, del percorso svolto in e-scooter (lista di liste)
- `matched`: variabile buleana, True se il map-matching ha avuto successo, altrimenti False. In caso assuma valore False, le coordinate riportate in `route` sono quelle originarie, precendenti al map-matching.
- `dates`: data (annp-mese-giorno), formato dt.date
- `month`: mese (integer)
- `year`: anno (integer)
- `edges_id`: lista contenente gli identificativi dei segmenti stradali ricavati da Valhalla
- `start_time`: timestamp di inizio corsa
- `end_time`: timestamp di fine corsa
- `distance`: distanza percorsa, calcolata con pyproj.Geod sui dati successivi al map-matching, per maggior precisione
- `speed`: velocità complessiva della corsa (in m/s)
- `speeds_for_edge`: lista contenente le velocità per ogni segmento stradale nella corsa, ricavata da Valhalla, espressa in km/h
- `street_names`: nomi delle strade o aree attraversate durante la corsa in e-scooter, ricavate da Valhalla. ID del segmento stradale, velocità per segmento e nome della strada presenti allo stesso indice nelle rispettive liste (dato un unique_id) sono da considerarsi informazioni corrispondenti.
- `start_hour`: ora di inizio corsa (integer)